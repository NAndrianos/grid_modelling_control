function Ybus = get_ybus(Z, n_buses, connections, shunt)
    % Initialize Ybus matrix with zeros
    Ybus = zeros(n_buses, n_buses);
    
    % Number of lines
    n_lines = length(Z);
    
    % Calculate admittances
    Y = 1 ./ Z;
    
    % Build the Y-bus matrix
    for k = 1:n_lines
        % Get the buses connected by the kth line
        i = connections(k, 1); % start point
        j = connections(k, 2); % end point
        
        % Update Ybus matrix
        Ybus(i, i) = Ybus(i, i) + Y(k);
        Ybus(j, j) = Ybus(j, j) + Y(k);
        Ybus(i, j) = Ybus(i, j) - Y(k);
        Ybus(j, i) = Ybus(j, i) - Y(k);

        % Add shunt admittance if it exists
        if shunt(k) ~= 0
            Ybus(i, i) = Ybus(i, i) + shunt(k); % Maybe divide shunt by 2
            Ybus(j, j) = Ybus(j, j) + shunt(k); % Same as above
        end
    end
end