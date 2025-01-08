import os

import matplotlib.pyplot as plt
import torch
from kan import KAN, LBFGS
from tqdm import tqdm
import numpy as np
import sympy
import copy


class KARHNN(KAN):

    def __init__(
        self,
        input_dim,
        width=None,
        grid=3,
        k=3,
        mult_arity=2,
        noise_scale=0.3,
        scale_base_mu=0.0,
        scale_base_sigma=1.0,
        base_fun="silu",
        symbolic_enabled=True,
        affine_trainable=False,
        grid_eps=0.02,
        grid_range=[-1, 1],
        sp_trainable=True,
        sb_trainable=True,
        seed=1,
        save_act=True,
        sparse_init=False,
        auto_save=True,
        first_init=True,
        ckpt_path="./model",
        state_id=0,
        round=0,
        device="cpu",
        assume_canonical_coords=True,
        field_type="solenoidal",
    ):
        super().__init__(
            width=width,
            grid=grid,
            k=k,
            mult_arity=mult_arity,
            noise_scale=noise_scale,
            scale_base_mu=scale_base_mu,
            scale_base_sigma=scale_base_sigma,
            base_fun=base_fun,
            symbolic_enabled=symbolic_enabled,
            affine_trainable=affine_trainable,
            grid_eps=grid_eps,
            grid_range=grid_range,
            sp_trainable=sp_trainable,
            sb_trainable=sb_trainable,
            seed=seed,
            save_act=save_act,
            sparse_init=sparse_init,
            auto_save=auto_save,
            first_init=first_init,
            ckpt_path=ckpt_path,
            state_id=state_id,
            round=round,
            device=device,
        )

        self.assume_canonical_coords = assume_canonical_coords
        self.input_dim = input_dim
        self.M = self.permutation_tensor(input_dim)  # Levi-Civita permutation tensor
        self.field_type = field_type

    def fit(
        self,
        dataset,
        opt="LBFGS",
        steps=100,
        log=1,
        lamb=0.0,
        lamb_l1=1.0,
        lamb_entropy=2.0,
        lamb_coef=0.0,
        lamb_coefdiff=0.0,
        update_grid=True,
        grid_update_num=10,
        loss_fn=None,
        lr=1.0,
        start_grid_update_step=-1,
        stop_grid_update_step=50,
        batch=-1,
        metrics=None,
        save_fig=False,
        in_vars=None,
        out_vars=None,
        beta=3,
        save_fig_freq=1,
        img_folder="./video",
        singularity_avoiding=False,
        y_th=1000.0,
        reg_metric="edge_forward_spline_n",
        display_metrics=None,
    ):
        """
        training

        Args:
        -----
            dataset : dic
                contains dataset['train_input'], dataset['train_label'], dataset['test_input'], dataset['test_label']
            opt : str
                "LBFGS" or "Adam"
            steps : int
                training steps
            log : int
                logging frequency
            lamb : float
                overall penalty strength
            lamb_l1 : float
                l1 penalty strength
            lamb_entropy : float
                entropy penalty strength
            lamb_coef : float
                coefficient magnitude penalty strength
            lamb_coefdiff : float
                difference of nearby coefficits (smoothness) penalty strength
            update_grid : bool
                If True, update grid regularly before stop_grid_update_step
            grid_update_num : int
                the number of grid updates before stop_grid_update_step
            start_grid_update_step : int
                no grid updates before this training step
            stop_grid_update_step : int
                no grid updates after this training step
            loss_fn : function
                loss function
            lr : float
                learning rate
            batch : int
                batch size, if -1 then full.
            save_fig_freq : int
                save figure every (save_fig_freq) steps
            singularity_avoiding : bool
                indicate whether to avoid singularity for the symbolic part
            y_th : float
                singularity threshold (anything above the threshold is considered singular and is softened in some ways)
            reg_metric : str
                regularization metric. Choose from {'edge_forward_spline_n', 'edge_forward_spline_u', 'edge_forward_sum', 'edge_backward', 'node_backward'}
            metrics : a list of metrics (as functions)
                the metrics to be computed in training
            display_metrics : a list of functions
                the metric to be displayed in tqdm progress bar

        Returns:
        --------
            results : dic
                results['train_loss'], 1D array of training losses (RMSE)
                results['test_loss'], 1D array of test losses (RMSE)
                results['reg'], 1D array of regularization
                other metrics specified in metrics

        Example
        -------
        >>> from kan import *
        >>> model = KAN(width=[2,5,1], grid=5, k=3, noise_scale=0.3, seed=2)
        >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
        >>> dataset = create_dataset(f, n_var=2)
        >>> model.fit(dataset, opt='LBFGS', steps=20, lamb=0.001);
        >>> model.plot()
        # Most examples in toturals involve the fit() method. Please check them for useness.
        """

        if lamb > 0.0 and not self.save_act:
            print("setting lamb=0. If you want to set lamb > 0, set self.save_act=True")

        old_save_act, old_symbolic_enabled = self.disable_symbolic_in_fit(lamb)

        pbar = tqdm(range(steps), desc="description", ncols=100)

        if loss_fn == None:
            loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2)
        else:
            loss_fn = loss_fn_eval = loss_fn

        grid_update_freq = int(stop_grid_update_step / grid_update_num)

        if opt == "Adam":
            optimizer = torch.optim.Adam(self.get_params(), lr=lr)
        elif opt == "LBFGS":
            optimizer = LBFGS(
                self.get_params(),
                lr=lr,
                history_size=10,
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-32,
                tolerance_change=1e-32,
                tolerance_ys=1e-32,
            )

        results = {}
        results["train_loss"] = []
        results["train_std"] = []
        results["test_loss"] = []
        results["test_std"] = []
        results["reg"] = []
        if metrics != None:
            for i in range(len(metrics)):
                results[metrics[i].__name__] = []

        if batch == -1 or batch > dataset["train_input"].shape[0]:
            batch_size = dataset["train_input"].shape[0]
            batch_size_test = dataset["test_input"].shape[0]
        else:
            batch_size = batch
            batch_size_test = batch

        global train_loss, reg_

        def closure():
            global train_loss, reg_, train_std
            optimizer.zero_grad()
            pred = self.time_derivative(
                dataset["train_input"][train_id],
                singularity_avoiding=singularity_avoiding,
                y_th=y_th,
            )
            train_loss = loss_fn(pred, dataset["train_label"][train_id])
            if self.save_act:
                if reg_metric == "edge_backward":
                    self.attribute()
                if reg_metric == "node_backward":
                    self.node_attribute()
                reg_ = self.get_reg(
                    reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff
                )
            else:
                reg_ = torch.tensor(0.0)
            objective = train_loss + lamb * reg_
            objective.backward()
            return objective

        if save_fig:
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)

        for curr_step in pbar:

            if curr_step == steps - 1 and old_save_act:
                self.save_act = True

            if save_fig and curr_step % save_fig_freq == 0:
                save_act = self.save_act
                self.save_act = True

            train_id = np.random.choice(
                dataset["train_input"].shape[0], batch_size, replace=False
            )
            test_id = np.random.choice(
                dataset["test_input"].shape[0], batch_size_test, replace=False
            )

            if (
                curr_step % grid_update_freq == 0
                and curr_step < stop_grid_update_step
                and update_grid
                and curr_step >= start_grid_update_step
            ):
                self.update_grid(dataset["train_input"][train_id])

            if opt == "LBFGS":
                optimizer.step(closure)

            if opt == "Adam":
                pred = self.time_derivative(
                    dataset["train_input"][train_id],
                    singularity_avoiding=singularity_avoiding,
                    y_th=y_th,
                )
                train_loss = loss_fn(pred, dataset["train_label"][train_id])
            
                if self.save_act:
                    if reg_metric == "edge_backward":
                        self.attribute()
                    if reg_metric == "node_backward":
                        self.node_attribute()
                    reg_ = self.get_reg(
                        reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff
                    )
                else:
                    reg_ = torch.tensor(0.0)
                loss = train_loss + lamb * reg_
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            test_loss = loss_fn_eval(
                self.time_derivative(dataset["test_input"][test_id]),
                dataset["test_label"][test_id],
            )

            if metrics != None:
                for i in range(len(metrics)):
                    results[metrics[i].__name__].append(metrics[i]().item())

            results["train_loss"].append(train_loss.cpu().detach().numpy())
            results["test_loss"].append(test_loss.cpu().detach().numpy())
            results["reg"].append(reg_.cpu().detach().numpy())

            if curr_step == steps - 1:
                train_dist = (dataset['train_label'][train_id] - self.time_derivative(dataset["train_input"][train_id]))**2
                results['train_std'].append(train_dist.std().item()/np.sqrt(train_dist.shape[0]))
                test_dist = (dataset['test_label'][test_id] - self.time_derivative(dataset["test_input"][test_id]))**2
                results['test_std'].append(test_dist.std().item()/np.sqrt(test_dist.shape[0]))
                
            if curr_step % log == 0:
                if display_metrics == None:
                    pbar.set_description(
                        "| train_loss: %.2e | test_loss: %.2e | reg: %.2e | "
                        % (
                            train_loss.cpu().detach().numpy(),
                            test_loss.cpu().detach().numpy(),
                            reg_.cpu().detach().numpy(),
                        )
                    )
                else:
                    string = ""
                    data = ()
                    for metric in display_metrics:
                        string += f" {metric}: %.2e |"
                        try:
                            results[metric]
                        except:
                            raise Exception(f"{metric} not recognized")
                        data += (results[metric][-1],)
                    pbar.set_description(string % data)

            if save_fig and curr_step % save_fig_freq == 0:
                self.plot(
                    folder=img_folder,
                    in_vars=in_vars,
                    out_vars=out_vars,
                    title="Step {}".format(curr_step),
                    beta=beta,
                )
                plt.savefig(
                    img_folder + "/" + str(curr_step) + ".jpg",
                    bbox_inches="tight",
                    dpi=200,
                )
                plt.close()
                self.save_act = save_act

        self.log_history("fit")
        # revert back to original state
        self.symbolic_enabled = old_symbolic_enabled
        return results

    def forward(self, x, singularity_avoiding=False, y_th=10.0):
        """
        forward pass

        Args:
        -----
            x : 2D torch.tensor
                inputs
            singularity_avoiding : bool
                whether to avoid singularity for the symbolic branch
            y_th : float
                the threshold for singularity

        Returns:
        --------
            None

        Example1
        --------
        >>> from kan import *
        >>> model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
        >>> x = torch.rand(100,2)
        >>> model(x).shape

        Example2
        --------
        >>> from kan import *
        >>> model = KAN(width=[1,1], grid=5, k=3, seed=0)
        >>> x = torch.tensor([[1],[-0.01]])
        >>> model.fix_symbolic(0,0,0,'log',fit_params_bool=False)
        >>> print(model(x))
        >>> print(model(x, singularity_avoiding=True))
        >>> print(model(x, singularity_avoiding=True, y_th=1.))
        """

        x = x[:, self.input_id.long()]
        assert x.shape[1] == self.width_in[0]

        # cache data
        self.cache_data = x

        self.acts = []  # shape ([batch, n0], [batch, n1], ..., [batch, n_L])
        self.acts_premult = []
        self.spline_preacts = []
        self.spline_postsplines = []
        self.spline_postacts = []
        self.acts_scale = []
        self.acts_scale_spline = []
        self.subnode_actscale = []
        self.edge_actscale = []
        # self.neurons_scale = []

        self.acts.append(x)  # acts shape: (batch, width[l])

        for l in range(self.depth):

            x_numerical, preacts, postacts_numerical, postspline = self.act_fun[l](x)
            # print(preacts, postacts_numerical, postspline)

            if self.symbolic_enabled == True:
                x_symbolic, postacts_symbolic = self.symbolic_fun[l](
                    x, singularity_avoiding=singularity_avoiding, y_th=y_th
                )
            else:
                x_symbolic = 0.0
                postacts_symbolic = 0.0

            x = x_numerical + x_symbolic

            if self.save_act:
                # save subnode_scale
                self.subnode_actscale.append(torch.std(x, dim=0).detach())

            # subnode affine transform
            x = self.subnode_scale[l][None, :] * x + self.subnode_bias[l][None, :]

            if self.save_act:
                postacts = postacts_numerical + postacts_symbolic

                # self.neurons_scale.append(torch.mean(torch.abs(x), dim=0))
                # grid_reshape = self.act_fun[l].grid.reshape(self.width_out[l + 1], self.width_in[l], -1)
                input_range = torch.std(preacts, dim=0) + 0.1
                output_range_spline = torch.std(
                    postacts_numerical, dim=0
                )  # for training, only penalize the spline part
                output_range = torch.std(
                    postacts, dim=0
                )  # for visualization, include the contribution from both spline + symbolic
                # save edge_scale
                self.edge_actscale.append(output_range)

                self.acts_scale.append((output_range / input_range).detach())
                self.acts_scale_spline.append(output_range_spline / input_range)
                self.spline_preacts.append(preacts.detach())
                self.spline_postacts.append(postacts.detach())
                self.spline_postsplines.append(postspline.detach())

                self.acts_premult.append(x.detach())

            # multiplication
            dim_sum = self.width[l + 1][0]
            dim_mult = self.width[l + 1][1]

            if self.mult_homo == True:
                for i in range(self.mult_arity - 1):
                    if i == 0:
                        x_mult = (
                            x[:, dim_sum :: self.mult_arity]
                            * x[:, dim_sum + 1 :: self.mult_arity]
                        )
                    else:
                        x_mult = x_mult * x[:, dim_sum + i + 1 :: self.mult_arity]

            else:
                for j in range(dim_mult):
                    acml_id = dim_sum + np.sum(self.mult_arity[l + 1][:j])
                    for i in range(self.mult_arity[l + 1][j] - 1):
                        if i == 0:
                            x_mult_j = x[:, [acml_id]] * x[:, [acml_id + 1]]
                        else:
                            x_mult_j = x_mult_j * x[:, [acml_id + i + 1]]

                    if j == 0:
                        x_mult = x_mult_j
                    else:
                        x_mult = torch.cat([x_mult, x_mult_j], dim=1)

            if self.width[l + 1][1] > 0:
                x = torch.cat([x[:, :dim_sum], x_mult], dim=1)

            # x = x + self.biases[l].weight
            # node affine transform
            x = self.node_scale[l][None, :] * x + self.node_bias[l][None, :]

            self.acts.append(x.detach())

        return x.split(1, 1)

    def time_derivative(
        self, x, t=None, separate_fields=False, singularity_avoiding=False, y_th=10.0
    ):
        """NEURAL HAMILTONIAN-STLE VECTOR FIELD"""
        kan_output = self.forward(
            x, singularity_avoiding=singularity_avoiding, y_th=y_th
        )  # traditional forward pass
        conservative_hamitonian, solenoidal_hamitonian = kan_output

        conservative_field = torch.zeros_like(
            x
        )  # start out with both components set to 0
        solenoidal_field = torch.zeros_like(x)

        if self.field_type != "solenoidal":
            dF1 = torch.autograd.grad(
                conservative_hamitonian.sum(), x, create_graph=True
            )[
                0
            ]  # gradients for conservative field
            conservative_field = dF1 @ torch.eye(*self.M.shape)

        if self.field_type != "conservative":
            dF2 = torch.autograd.grad(
                solenoidal_hamitonian.sum(), x, create_graph=True
            )[
                0
            ]  # gradients for solenoidal field
            solenoidal_field = dF2 @ self.M.t()

        if separate_fields:
            return [conservative_field, solenoidal_field]

        return conservative_field + solenoidal_field

    def permutation_tensor(self, n):
        M = None
        if self.assume_canonical_coords:
            M = torch.eye(n)
            M = torch.cat([M[n // 2 :], -M[: n // 2]])
        else:
            """Constructs the Levi-Civita permutation tensor"""
            M = torch.ones(n, n)  # matrix of ones
            M *= 1 - torch.eye(n)  # clear diagonals
            M[::2] *= -1  # pattern of signs
            M[:, ::2] *= -1

            for i in range(n):  # make asymmetric
                for j in range(i + 1, n):
                    M[i, j] *= -1
        return M

    # Change return type from MultKAN to KANHNN model.
    def prune_node(
        self, threshold=1e-2, mode="auto", active_neurons_id=None, log_history=True
    ):
        """
        pruning nodes

        Args:
        -----
            threshold : float
                if the attribution score of a neuron is below the threshold, it is considered dead and will be removed
            mode : str
                'auto' or 'manual'. with 'auto', nodes are automatically pruned using threshold. with 'manual', active_neurons_id should be passed in.

        Returns:
        --------
            pruned network : MultKAN

        Example
        -------
        >>> from kan import *
        >>> model = KAN(width=[2,5,1], grid=5, k=3, noise_scale=0.3, seed=2)
        >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
        >>> dataset = create_dataset(f, n_var=2)
        >>> model.fit(dataset, opt='LBFGS', steps=20, lamb=0.001);
        >>> model = model.prune_node()
        >>> model.plot()
        """
        if self.acts == None:
            self.get_act()

        mask_up = [torch.ones(self.width_in[0], device=self.device)]
        mask_down = []
        active_neurons_up = [list(range(self.width_in[0]))]
        active_neurons_down = []
        num_sums = []
        num_mults = []
        mult_arities = [[]]

        if active_neurons_id != None:
            mode = "manual"

        for i in range(len(self.acts_scale) - 1):

            mult_arity = []

            if mode == "auto":
                self.attribute()
                overall_important_up = self.node_scores[i + 1] > threshold

            elif mode == "manual":
                overall_important_up = torch.zeros(
                    self.width_in[i + 1], dtype=torch.bool, device=self.device
                )
                overall_important_up[active_neurons_id[i]] = True

            num_sum = torch.sum(overall_important_up[: self.width[i + 1][0]])
            num_mult = torch.sum(overall_important_up[self.width[i + 1][0] :])
            if self.mult_homo == True:
                overall_important_down = torch.cat(
                    [
                        overall_important_up[: self.width[i + 1][0]],
                        (
                            overall_important_up[self.width[i + 1][0] :][
                                None, :
                            ].expand(self.mult_arity, -1)
                        ).T.reshape(
                            -1,
                        ),
                    ],
                    dim=0,
                )
            else:
                overall_important_down = overall_important_up[: self.width[i + 1][0]]
                for j in range(overall_important_up[self.width[i + 1][0] :].shape[0]):
                    active_bool = overall_important_up[self.width[i + 1][0] + j]
                    arity = self.mult_arity[i + 1][j]
                    overall_important_down = torch.cat(
                        [
                            overall_important_down,
                            torch.tensor([active_bool] * arity).to(self.device),
                        ]
                    )
                    if active_bool:
                        mult_arity.append(arity)

            num_sums.append(num_sum.item())
            num_mults.append(num_mult.item())

            mask_up.append(overall_important_up.float())
            mask_down.append(overall_important_down.float())

            active_neurons_up.append(torch.where(overall_important_up == True)[0])
            active_neurons_down.append(torch.where(overall_important_down == True)[0])

            mult_arities.append(mult_arity)

        active_neurons_down.append(list(range(self.width_out[-1])))
        mask_down.append(torch.ones(self.width_out[-1], device=self.device))

        if self.mult_homo == False:
            mult_arities.append(self.mult_arity[-1])

        self.mask_up = mask_up
        self.mask_down = mask_down

        # update act_fun[l].mask up
        for l in range(len(self.acts_scale) - 1):
            for i in range(self.width_in[l + 1]):
                if i not in active_neurons_up[l + 1]:
                    self.remove_node(l + 1, i, mode="up", log_history=False)

            for i in range(self.width_out[l + 1]):
                if i not in active_neurons_down[l]:
                    self.remove_node(l + 1, i, mode="down", log_history=False)

        model2 = KARHNN(
            input_dim=self.input_dim,
            width=copy.deepcopy(self.width),
            grid=self.grid,
            k=self.k,
            base_fun=self.base_fun_name,
            mult_arity=self.mult_arity,
            ckpt_path=self.ckpt_path,
            auto_save=True,
            first_init=False,
            state_id=self.state_id,
            round=self.round,
        ).to(self.device)
        model2.load_state_dict(self.state_dict())

        width_new = [self.width[0]]

        for i in range(len(self.acts_scale)):

            if i < len(self.acts_scale) - 1:
                num_sum = num_sums[i]
                num_mult = num_mults[i]
                model2.node_bias[i].data = model2.node_bias[i].data[
                    active_neurons_up[i + 1]
                ]
                model2.node_scale[i].data = model2.node_scale[i].data[
                    active_neurons_up[i + 1]
                ]
                model2.subnode_bias[i].data = model2.subnode_bias[i].data[
                    active_neurons_down[i]
                ]
                model2.subnode_scale[i].data = model2.subnode_scale[i].data[
                    active_neurons_down[i]
                ]
                model2.width[i + 1] = [num_sum, num_mult]

                model2.act_fun[i].out_dim_sum = num_sum
                model2.act_fun[i].out_dim_mult = num_mult

                model2.symbolic_fun[i].out_dim_sum = num_sum
                model2.symbolic_fun[i].out_dim_mult = num_mult

                width_new.append([num_sum, num_mult])

            model2.act_fun[i] = model2.act_fun[i].get_subset(
                active_neurons_up[i], active_neurons_down[i]
            )
            model2.symbolic_fun[i] = self.symbolic_fun[i].get_subset(
                active_neurons_up[i], active_neurons_down[i]
            )

        model2.cache_data = self.cache_data
        model2.acts = None

        width_new.append(self.width[-1])
        model2.width = width_new

        if self.mult_homo == False:
            model2.mult_arity = mult_arities

        if log_history:
            self.log_history("prune_node")
            model2.state_id += 1

        return model2

    def symbolic_formula(self, var=None, normalizer=None, output_normalizer=None):
        """
        get symbolic formula

        Args:
        -----
            var : None or a list of sympy expression
                input variables
            normalizer : [mean, std]
            output_normalizer : [mean, std]

        Returns:
        --------
            None

        Example
        -------
        >>> from kan import *
        >>> model = KAN(width=[2,1,1], grid=5, k=3, noise_scale=0.0, seed=0)
        >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]])+x[:,[1]]**2)
        >>> dataset = create_dataset(f, n_var=3)
        >>> model.fit(dataset, opt='LBFGS', steps=20, lamb=0.001);
        >>> model.auto_symbolic()
        >>> model.symbolic_formula()[0][0]
        """

        symbolic_acts = []
        symbolic_acts_premult = []
        x = []

        def ex_round(ex1, n_digit):
            ex2 = ex1
            for a in sympy.preorder_traversal(ex1):
                if isinstance(a, sympy.Float):
                    ex2 = ex2.subs(a, round(a, n_digit))
            return ex2

        # define variables
        if var == None:
            for ii in range(1, self.width[0][0] + 1):
                x.append(sympy.Symbol(f"x_{ii}"))
                # exec(f"x{ii} = sympy.Symbol('x_{ii}')")
                # print(x1)
                # exec(f"x.append(x{ii})")
        elif isinstance(var[0], sympy.Expr):
            x = var
        else:
            x = [sympy.symbols(var_) for var_ in var]

        x0 = x

        if normalizer != None:
            mean = normalizer[0]
            std = normalizer[1]
            x = [(x[i] - mean[i]) / std[i] for i in range(len(x))]

        symbolic_acts.append(x)

        simplify = True
        for l in range(len(self.width_in) - 1):
            num_sum = self.width[l + 1][0]
            num_mult = self.width[l + 1][1]
            y = []
            for j in range(self.width_out[l + 1]):
                yj = 0.0
                for i in range(self.width_in[l]):
                    a, b, c, d = self.symbolic_fun[l].affine[j, i]
                    sympy_fun = self.symbolic_fun[l].funs_sympy[j][i]
                    try:
                        yj += c * sympy_fun(a * x[i] + b) + d
                    except:
                        print(
                            "make sure all activations need to be converted to symbolic formulas first!"
                        )
                        return
                yj = self.subnode_scale[l][j] * yj + self.subnode_bias[l][j]
                if simplify == True:
                    y.append(sympy.simplify(yj))
                else:
                    y.append(yj)

            symbolic_acts_premult.append(y)

            mult = []
            for k in range(num_mult):
                if isinstance(self.mult_arity, int):
                    mult_arity = self.mult_arity
                else:
                    mult_arity = self.mult_arity[l + 1][k]
                for i in range(mult_arity - 1):
                    if i == 0:
                        mult_k = y[num_sum + 2 * k] * y[num_sum + 2 * k + 1]
                    else:
                        mult_k = mult_k * y[num_sum + 2 * k + i + 1]
                mult.append(mult_k)

            y = y[:num_sum] + mult

            for j in range(self.width_in[l + 1]):
                y[j] = self.node_scale[l][j] * y[j] + self.node_bias[l][j]

            x = y
            symbolic_acts.append(x)

        if output_normalizer != None:
            output_layer = symbolic_acts[-1]
            means = output_normalizer[0]
            stds = output_normalizer[1]

            assert len(output_layer) == len(
                means
            ), "output_normalizer does not match the output layer"
            assert len(output_layer) == len(
                stds
            ), "output_normalizer does not match the output layer"

            output_layer = [
                (output_layer[i] * stds[i] + means[i]) for i in range(len(output_layer))
            ]
            symbolic_acts[-1] = output_layer

        self.symbolic_acts = [
            [symbolic_acts[l][i] for i in range(len(symbolic_acts[l]))]
            for l in range(len(symbolic_acts))
        ]
        self.symbolic_acts_premult = [
            [symbolic_acts_premult[l][i] for i in range(len(symbolic_acts_premult[l]))]
            for l in range(len(symbolic_acts_premult))
        ]

        out_dim = len(symbolic_acts[-1])
        # return [symbolic_acts[-1][i] for i in range(len(symbolic_acts[-1]))], x0

        if simplify:
            return [symbolic_acts[-1][i] for i in range(len(symbolic_acts[-1]))], x0
        else:
            return [symbolic_acts[-1][i] for i in range(len(symbolic_acts[-1]))], x0

    def refine(self, new_grid):
        """
        grid refinement

        Args:
        -----
            new_grid : init
                the number of grid intervals after refinement

        Returns:
        --------
            a refined model : MultKAN

        Example
        -------
        >>> from kan import *
        >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        >>> model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
        >>> print(model.grid)
        >>> x = torch.rand(100,2)
        >>> model.get_act(x)
        >>> model = model.refine(10)
        >>> print(model.grid)
        checkpoint directory created: ./model
        saving model version 0.0
        5
        saving model version 0.1
        10
        """

        model_new = KARHNN(
            input_dim=self.input_dim,
            width=self.width,
            grid=new_grid,
            k=self.k,
            mult_arity=self.mult_arity,
            base_fun=self.base_fun_name,
            symbolic_enabled=self.symbolic_enabled,
            affine_trainable=self.affine_trainable,
            grid_eps=self.grid_eps,
            grid_range=self.grid_range,
            sp_trainable=self.sp_trainable,
            sb_trainable=self.sb_trainable,
            ckpt_path=self.ckpt_path,
            auto_save=True,
            first_init=False,
            state_id=self.state_id,
            round=self.round,
            device=self.device,
        )

        model_new.initialize_from_another_model(self, self.cache_data)
        model_new.cache_data = self.cache_data
        model_new.grid = new_grid

        self.log_history("refine")
        model_new.state_id += 1

        return model_new.to(self.device)
