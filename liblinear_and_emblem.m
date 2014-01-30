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
    This function intializes a classifier (w, b) using liblinear and then executes the function emblem.
    For description of arguments, please read emblem.m.
    This function requires MATLAB interface to LIBLINEAR.
%}


function [w, b] = liblinear_and_emblem(Xl, y, Xu, assumption, lambda, c, use_pcg, use_fn_handle, pcg_tol, em_tol, print_status, mean_upper_bound, mean_lower_bound, use_precond)

dimension = size(Xl, 1);
inv_lambda = 1 / lambda;
option_string = sprintf('-s 2 -B 1 -q -c %.5f', inv_lambda);
model = train(y, Xl', option_string);
w = model.w(1:dimension)';
b = model.w(dimension + 1);
if model.Label(1) == -1
    w = -w;
    b = -b;
end

[w, b] = emblem(Xl, y, Xu, assumption, w, b, lambda, c, use_pcg, use_fn_handle, pcg_tol, em_tol, print_status, mean_upper_bound, mean_lower_bound, use_precond);


