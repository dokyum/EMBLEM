%{
Copyright [2014] [Do-kyum Kim, Matthew Der and Lawrence K. Saul]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

%}

%{
Desc
    This is the core function that estimates a classifier (w, b) using the EMBLEM algorithm.

Input
    Xl: labeled examples
    y: labels
    Xu: unlabeled examples
    assumption: 1 = EMBLEM_{ss}, 2 = EMBLEM_{ss}^{bal}
    w: initial value for w
    b: initial value for b
    lambda: regularization parameter. lambda > 0. We used lambda = 1.
    c: overrelaxation parameter. 0 <= c <= 1, or c = -1 means "use c = 1 first, get convergence, then use c = 0".
       We used c = -1.
    use_pcg: 1 = use pcg to solve least-squares problems, 0 = use matrix inversion
    use_fn_handle: 1 = use function handle to avoid computing the matrix A in pcg. 0 = precompute the matrix A.
    pcg_tol: tolerance for pcg. We used pcg_tol = 1e-6.
    em_tol: tolerance for convergence of the EM algorithm. We used em_tol = 1e-4.
    print_status: 1 = print status at each iteration, 0 = do not print
    mean_upper_bound: upper bound on the mean of labels for the unlabeled examples (used for EMBLEM_{ss}^{bal})
    mean_lower_bound: lower bound ...
    use_precond: 1 = use preconditioner for pcg, 0 = do not use

Output
    w
    b
%}

function [w, b] = emblem(Xl, y, Xu, assumption, w, b, lambda, c, use_pcg, use_fn_handle, pcg_tol, em_tol, print_status, mean_upper_bound, mean_lower_bound, use_precond)

[d,L] = size(Xl);
U = size(Xu,2);
N = L+U;

% parameter checks
assert(issparse(Xl));
assert(issparse(Xu));
assert(isnumeric(w),'weight vector w must be numeric!');
assert(isnumeric(b),'scalar bias b must be numeric!');
assert(size(y, 1) == L && size(y, 2) == 1, 'size of y is inappropriate!');
assert(size(Xu,1)==d,'labeled data Xl and unlabeled data Xu must have the same dimensionality!');
assert(sum(size(w)==[d,1])==2,'weight vector w should be sized as d x 1!');
assert(isscalar(b),'bias b must be scalar (size 1 x 1)!');
assert(assumption == 1 || assumption == 2, 'assumption should be one of {1, 2}!');
assert(c == -1 || (c >= 0 && c <= 1), 'c is not set correctly!');
assert(mean_upper_bound <= 1 && mean_upper_bound >= -1, 'mean_upper_bound should be in [-1, 1]');
assert(mean_lower_bound <= 1 && mean_lower_bound >= -1, 'mean_lower_bound should be in [-1, 1]');

% other parameters
maxIter = 25000; % maximum # iterations

% other initializations, one-time computations
X = horzcat(Xl, Xu); % all data

if use_pcg
    if ~use_fn_handle
        Xreg = X*X' + lambda*speye(d);
        if use_precond
           M1 = ichol(Xreg);
        end
    end
else
    if d <= N
        XregInv = inv(X*X' + lambda*speye(d));
    else
        XregInv = (speye(d) - X * inv(X' * X + lambda * speye(N)) * X') / lambda;
    end
    XregInv_times_X = XregInv*X;
    clear XregInv;
end

if c == -1
    actual_c = 1;
else
    actual_c = c;
end
last_actual_c = 0;

sqrt2overpi = sqrt(2/pi);
posterior_factor_l = sqrt2overpi*y;
sqrt2 = sqrt(2);
log2 = log(2);
log_pi_div_2 = log(pi) / 2;
Llog2 = -L * log(2);
sqrt1over2pi = 1 / sqrt(2 * pi);
Uoversqrt2 = U / sqrt(2);
twooversqrtpi = 2 / sqrt(pi);
Usqr = U * U;

prevLL = -Inf;

for iter=1:maxIter
    % compute prior means
    % labeled
    prior_mean_l = Xl'*w+b; % E[z|x] = w.x+b
    arg = (1-y.*prior_mean_l)/sqrt2;

    % unlabeled
    prior_mean_u = Xu'*w+b; % E[z|x] = w.x+b
    
    % Compute posterior by large margin
    % labeled
    posterior_mean_l = prior_mean_l + posterior_factor_l./erfcx(arg); % E[z|x,y], y={+1,-1}
    
    % unlabeled
    arg_pos = (1-prior_mean_u)/sqrt2;
    arg_neg = (1+prior_mean_u)/sqrt2;

    prob_pos = erfc(arg_pos) / 2;
    prob_neg = erfc(arg_neg) / 2;
    prob_sum = prob_pos + prob_neg;
    rho_pos = prob_pos ./ prob_sum;
    rho_neg = prob_neg ./ prob_sum;
    
    expectation_z_pos = prior_mean_u + sqrt2overpi./erfcx(arg_pos); % E[z|x,y=+1]
    expectation_z_neg = prior_mean_u - sqrt2overpi./erfcx(arg_neg); % E[z|x,y=-1]

    posterior_mean_u = rho_pos .* expectation_z_pos + rho_neg .* expectation_z_neg;

    if assumption == 2
        intermediate_mean_y = rho_pos - rho_neg;
        intermediate_var_y = 1 - power(intermediate_mean_y, 2);

        intermediate_mean_y_sum = sum(intermediate_mean_y);
        intermediate_var_y_sum = sum(intermediate_var_y);
        intermediate_var_y_sum_sqrt = sqrt(intermediate_var_y_sum);

        arg_star_U = (U * mean_upper_bound - intermediate_mean_y_sum) / intermediate_var_y_sum_sqrt / sqrt2;
        arg_star_L = (U * mean_lower_bound - intermediate_mean_y_sum) / intermediate_var_y_sum_sqrt / sqrt2;

        derivative_prob_pos = sqrt1over2pi * exp(-power(arg_pos, 2));
        derivative_prob_neg = -sqrt1over2pi * exp(-power(arg_neg, 2));
        derivative_intermediate_mean = 2 * (rho_neg .* derivative_prob_pos - rho_pos .* derivative_prob_neg) ./ prob_sum;
        derivative_arg_star_U = derivative_intermediate_mean * (arg_star_U - Uoversqrt2) / intermediate_var_y_sum_sqrt / U;
        derivative_arg_star_L = derivative_intermediate_mean * (arg_star_L - Uoversqrt2) / intermediate_var_y_sum_sqrt / U;

        prob_star = (erf(arg_star_U) - erf(arg_star_L)) / 2;

        if prob_star > 0
            log_prob_star = log(prob_star);
            derivative_erf_U = twooversqrtpi * exp(-power(arg_star_U, 2)) .* derivative_arg_star_U;
            derivative_erf_L = twooversqrtpi * exp(-power(arg_star_L, 2)) .* derivative_arg_star_L;
            delta_u = (derivative_erf_U - derivative_erf_L) / 2 / prob_star;
        else
            if arg_star_L > 0
                log_prob_star = -log2 - power(arg_star_L, 2) - log(arg_star_L) - log_pi_div_2;
                delta_u = -derivative_arg_star_L * (2 * arg_star_L + 1 / arg_star_L);
            elseif arg_star_U < 0
                log_prob_star = -log2 - power(arg_star_U, 2) - log(-arg_star_U) - log_pi_div_2;
                delta_u = -derivative_arg_star_U * (2 * arg_star_U + 1 / arg_star_U);
            else
                assert(0);
            end
        end

        posterior_mean_u = posterior_mean_u + delta_u;
    end

    z = vertcat(posterior_mean_l, posterior_mean_u);
       
    % Compute likelihood

    % For labeled 
    LL = sum(log(erfc(arg))) + Llog2 - lambda*(w'*w)/2;

    % For unlabeled
    LL = LL + sum(log(prob_sum));

    % For mean constraint
    if assumption == 2
        LL = LL + log_prob_star;
    end

    deltaLL = (LL-prevLL)/abs(prevLL);
    
    % PRINT STATUS UPDATE
    if mod(iter, print_status) == 0
        fprintf('ITER %d:\n',iter);
        fprintf('  Log-likelihood  = %f\n',LL);
        fprintf('    LL delta      = %f\n',deltaLL);
        fprintf('  actual_c        = %f\n',last_actual_c);
        fprintf('  norm(w)^2       = %f\n',w'*w);
        fprintf('  mean_yu         = %f\n',mean(sign(Xu'*w+b)));
    end
    
    % converged?
    if iter > 2 && deltaLL < em_tol
        if c == -1
            if actual_c > 0
                actual_c = 0;
            else
                break;
            end
        else
            break;
        end
    end    
    prevLL = LL;
    
    % M-step: UPDATE MODEL PARAMETERS (w,b)
    prev_w = w;

    % SOLVE:  Xreg*w = X*(z-b)
    if use_pcg
        if use_fn_handle == 1
            [w, flag] = pcg(@computeXregw,X*(z-b),pcg_tol,1000000,[],[],prev_w);
            assert(flag == 0);
        else
            if use_precond
                [w, flag] = pcg(Xreg,X*(z-b),pcg_tol,1000000,M1,M1',prev_w);
            else
                [w, flag] = pcg(Xreg,X*(z-b),pcg_tol,1000000,[],[],prev_w);
            end
            assert(flag == 0);
        end
    else
        w = XregInv_times_X*(z-b);
    end
    w = (1+actual_c)*w - actual_c*prev_w; % successive overrelaxation update
    last_actual_c = actual_c;
    b = mean(z - X' * w);
end

    % function to compute Xreg*w
    function Xregw = computeXregw(w_)
        Xregw = X*(X'*w_)+lambda*w_;
    end

end
