from PreProsses import PreProcess
from Model import ModelCreator


# train, target = PreProcess('Travel_data_shifted.xlsx').process_input_data()
# ModelCreator(train, target, 'logistic').train_model(plot_input=True)
# ModelCreator(train, target, 'sgd').train_model()
# ModelCreator(train, target, 'tree').train_model()
# ModelCreator(train, target, 'linear').train_model()
# ModelCreator(train, target, 'knn').train_model()
# ModelCreator(train, target, 'lasso').train_model()
# ModelCreator(train, target, 'ridge').train_model()
# ModelCreator(train, target, 'bayesian').train_model()
# print()
# print()

# train, target = PreProcess('Travel_data.xlsx').process_input_data()
# ModelCreator(train, target, 'logistic').train_model()
# ModelCreator(train, target, 'sgd').train_model()
# ModelCreator(train, target, 'tree').train_model()
# ModelCreator(train, target, 'linear').train_model()
# ModelCreator(train, target, 'knn').train_model()
# ModelCreator(train, target, 'lasso').train_model()
# ModelCreator(train, target, 'ridge').train_model()
# ModelCreator(train, target, 'bayesian').train_model()
# print()
# print()

train, target = PreProcess('Travel_data_full_column.xlsx').process_input_data()
ModelCreator(train, target, 'logistic').train_model(plot_input=True)
ModelCreator(train, target, 'sgd').train_model()
ModelCreator(train, target, 'tree').train_model()
ModelCreator(train, target, 'linear').train_model()
ModelCreator(train, target, 'knn').train_model()
ModelCreator(train, target, 'lasso').train_model()
ModelCreator(train, target, 'ridge').train_model()
ModelCreator(train, target, 'bayesian').train_model()
print()
print()

# train, target = PreProcess('Travel_data_full_column.xlsx').process_input_data()
# ModelCreator(train, target, 'logistic', test_split_available=True).train_model(plot_input=True)
# ModelCreator(train, target, 'sgd', test_split_available=True).train_model()
# ModelCreator(train, target, 'tree', test_split_available=True).train_model()
# ModelCreator(train, target, 'linear', test_split_available=True).train_model()
# ModelCreator(train, target, 'knn', test_split_available=True).train_model()
# ModelCreator(train, target, 'lasso', test_split_available=True).train_model()
# ModelCreator(train, target, 'ridge', test_split_available=True).train_model()
# ModelCreator(train, target, 'bayesian', test_split_available=True).train_model()
# print()
# print()