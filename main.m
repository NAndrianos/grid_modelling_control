clear all
close all
clc

% Import and add correct path
addpath("C:\Users\winst\OneDrive\Documents\casadi-3.6.4-windows64-matlab2018b")
import casadi.*

% Define the inputs
Z = [0.04+0.2i, 0.0134+0.0605i, 0.02+0.04i];
n = 4; % number of busses
connections = [1 2; 2 3; 2 4];
shunt = [0, 2.25i, 0];  % shunt admittance values for each line

% Calculate the Ybus
% Laplacian_mat = get_laplacian(Z, n, connections, shunt); Same as Y_bus
Y_bus = get_ybus(Z, n, connections, shunt);

% Display the result
disp('Ybus matrix:');
disp(Y_bus);

% Define real and imaginary parts for voltages as symbolic variables
% NOTE in CasADi: V = V_real + 1i * V_imag we cannot define a complex
% variable like this. We need manually seperate real and imaginary
V_real = SX.sym('V_real', n);
V_imag = SX.sym('V_imag', n);

G_bus = real(Y_bus);
B_bus = imag(Y_bus);

% Compute real power P symbolic equation
P_eq = V_real .* (G_bus * V_real - B_bus * V_imag) + V_imag .* (B_bus * V_real + G_bus * V_imag);

% Compute reactive power Q symbolic equation
Q_eq = V_real .* (-B_bus * V_real - G_bus * V_imag) + V_imag .* (G_bus * V_real - B_bus * V_imag);

P = SX.sym('P', n);
Q = SX.sym('Q', n);

% Combine decision variables into a single vector
x = [P; Q; V_real; V_imag];

% Power balance equality constraints (without loops)
g = [];

% For all buses, enforce that the computed power equals the specified power
g = [g; P - P_eq];  % Active power balance
g = [g; Q - Q_eq];  % Reactive power balance


