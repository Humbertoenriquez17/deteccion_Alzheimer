clc; clear all; close all;

global X_train y_train X_sample y_sample y_min y_max resultado_texto;

data_negativo_train = csvread('../data/Dataset_negativo_Train.csv', 0, 0);
data_positivo_train = csvread('../data/Dataset_positivo_Train.csv', 0, 0);

data_negativo_test = csvread('../data/Dataset_negativo_Test.csv', 0, 0);
data_positivo_test = csvread('../data/Dataset_positivo_Test.csv', 0, 0);

X_train_negativo = data_negativo_train(1:532, :);
X_train_positivo = data_positivo_train(1:532, :);

X_train = [X_train_negativo; X_train_positivo];
y_train = [zeros(size(X_train_negativo, 1), 1); ones(size(X_train_positivo, 1), 1)];

X_sample = [data_negativo_test; data_positivo_test];
y_sample = [zeros(size(data_negativo_test, 1), 1); ones(size(data_positivo_test, 1), 1)];


y_min = 0;
y_max = 1;

fig = figure('Name', 'Clasificador Bayesiano', 'NumberTitle', 'off', ...
             'Position', [100, 100, 400, 300]);

btn_seleccionar = uicontrol('Style', 'pushbutton', 'String', 'Seleccionar Fila', ...
                            'Position', [150, 180, 100, 40], ...
                            'Callback', @seleccionar_y_clasificar);

resultado_texto = uicontrol('Style', 'text', 'String', '', ...
                            'Position', [100, 100, 200, 40], ...
                            'FontSize', 12);

function seleccionar_y_clasificar(~, ~)
    global X_train y_train X_sample y_sample y_min y_max resultado_texto;

    fila_muestra = inputdlg('Introduce el número de fila de la muestra (1 a 456):', ...
                            'Seleccionar Fila', [1 50]);
    fila_muestra = str2double(fila_muestra);

    if fila_muestra < 1 || fila_muestra > size(X_sample, 1)
        errordlg('El número de fila está fuera del rango válido.', 'Error');
        return;
    end

    muestra = X_sample(fila_muestra, :);

    X_normalizado = normalizar_datos([X_train; muestra], y_min, y_max);
    muestra_normalizada = X_normalizado(end, :);

    clase_predicha = clasificar_bayesiano(X_normalizado(1:end-1, :), y_train, muestra_normalizada);

    if clase_predicha == 0
        set(resultado_texto, 'String', 'Clase: 0 (Sin Alzheimer)');
    elseif clase_predicha == 1
        set(resultado_texto, 'String', 'Clase: 1 (Con Alzheimer)');
    else
        set(resultado_texto, 'String', 'Clase desconocida');
    end

    calcular_metricas()
end

function X_normalizado = normalizar_datos(X, y_min, y_max)
    X_min = min(X, [], 1);
    X_max = max(X, [], 1);
    X_normalizado = (y_max - y_min) * (X - X_min) ./ (X_max - X_min + eps) + y_min;
end

function clase = clasificar_bayesiano(X, y, muestra)
    clases = unique(y);
    num_clases = length(clases);
    prob_clase = zeros(num_clases, 1);
    for i = 1:num_clases
        indices = (y == clases(i));
        X_clase = X(indices, :);
        media = mean(X_clase, 1);
        varianza = var(X_clase, 0, 1) + eps;
        prob = -0.5 * sum(((muestra - media).^2) ./ varianza) - 0.5 * sum(log(2 * pi * varianza));
        prob_clase(i) = prob;
    end
    [~, idx] = max(prob_clase);
    clase = clases(idx);
end

function calcular_metricas()
    global X_train y_train X_sample y_sample y_min y_max;

    X_total = [X_train; X_sample];
    X_total_normalizado = normalizar_datos(X_total, y_min, y_max);

    X_train_normalizado = X_total_normalizado(1:size(X_train, 1), :);
    X_sample_normalizado = X_total_normalizado(size(X_train, 1)+1:end, :);

    y_pred = zeros(size(y_sample));
    for i = 1:size(X_sample_normalizado, 1)
        y_pred(i) = clasificar_bayesiano(X_train_normalizado, y_train, X_sample_normalizado(i, :));
    end

    TP = sum((y_pred == 1) & (y_sample == 1));
    TN = sum((y_pred == 0) & (y_sample == 0));
    FP = sum((y_pred == 1) & (y_sample == 0));
    FN = sum((y_pred == 0) & (y_sample == 1));

    accuracy = (TP + TN) / (TP + TN + FP + FN);
    precision = TP / (TP + FP + eps);
    recall = TP / (TP + FN + eps);
    f1_score = 2 * (precision * recall) / (precision + recall + eps);

    fprintf('=== MÉTRICAS DE DESEMPEÑO ===\n');
    fprintf('Precision (Predicciones positivas correctas): %.2f\n', precision);
    fprintf('Exactitud (sinónimo de Accuracy): %.2f\n', accuracy);
    fprintf('Recall (Sensibilidad): %.2f\n', recall);
    fprintf('F1-Score: %.2f\n', f1_score);
end

