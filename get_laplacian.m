function laplacian_matrix = get_laplacian(Z, n_buses, connections, shunt)
    % Initialize the adjacency matrix
    adj_matrix = zeros(n_buses);

    % Calculate the weights (admittance) from the impedance values
    weights = 1 ./ Z;
    
    % Fill the adjacency matrix with weights (admittances)
    for k = 1:length(weights)
        i = connections(k, 1);
        j = connections(k, 2);
        adj_matrix(i, j) = weights(k);
        adj_matrix(j, i) = weights(k); % Symmetric for undirected graph
    end

    % Add shunt admittances to the diagonal of the adjacency matrix
    % Note we can't just add the shunt like we do for Y-bus
    % for k = 1:length(weights)
    %     i = connections(k, 1);
    %     j = connections(k, 2);
    %     adj_matrix(i, i) = adj_matrix(i, i) + shunt(k);
    %     adj_matrix(j, j) = adj_matrix(j, j) + shunt(k);
    % end

    % Compute the degree matrix (sum of weights per node)
    degree_matrix = diag(sum(adj_matrix, 2));

    % Compute the Laplacian matrix
    laplacian_matrix = degree_matrix - adj_matrix;
end