pkg load statistics;

neg_train = csvread('../data/Dataset_negativo_Train.csv');
pos_train = csvread('../data/Dataset_positivo_Train.csv');
neg_test = csvread('../data/Dataset_negativo_Test.csv');
pos_test = csvread('../data/Dataset_positivo_Test.csv');

neg_train_labels = zeros(size(neg_train, 1), 1);
pos_train_labels = ones(size(pos_train, 1), 1);
neg_test_labels = zeros(size(neg_test, 1), 1);
pos_test_labels = ones(size(pos_test, 1), 1);

train_data = [neg_train; pos_train];
train_labels = [neg_train_labels; pos_train_labels];
test_data = [neg_test; pos_test];
test_labels = [neg_test_labels; pos_test_labels];

normalize = @(x) (x - mean(x)) ./ std(x);
train_data = normalize(train_data);
test_data = normalize(test_data);

model = svmtrain(train_labels, train_data, '-t 0 -c 0.1');

[predictions, accuracy, ~] = svmpredict(test_labels, test_data, model);

tp = sum((predictions == 1) & (test_labels == 1));
tn = sum((predictions == 0) & (test_labels == 0));
fp = sum((predictions == 1) & (test_labels == 0));
fn = sum((predictions == 0) & (test_labels == 1));

precision = tp / (tp + fp);
recall = tp / (tp + fn);
f1_score = 2 * (precision * recall) / (precision + recall);
accuracy = (tp + tn) / (tp + tn + fp + fn);

fprintf('Precisión: %.2f\n', precision);
fprintf('Exactitud: %.2f\n', accuracy);
fprintf('F1 Score: %.2f\n', f1_score);
fprintf('Recall: %.2f\n', recall);

row_number = input('Ingrese el número de fila del archivo de prueba: ');

if row_number > 0 && row_number <= size(test_data, 1)
    test_sample = test_data(row_number, :);
    actual_label = test_labels(row_number);
    predicted_label = svmpredict(actual_label, test_sample, model);

    fprintf('Etiqueta real: %d\n', actual_label);
    fprintf('Etiqueta predicha: %d\n', predicted_label);

    if actual_label == predicted_label
        fprintf('La predicción es correcta.\n');
    else
        fprintf('La predicción es incorrecta.\n');
    end
else
    fprintf('Número de fila inválido.\n');
end
