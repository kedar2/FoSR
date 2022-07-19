import argparse
import ast

def get_args_from_input():
	parser = argparse.ArgumentParser(description='modify network parameters')

	parser.add_argument('--learning_rate', metavar='', type=float, help='learning rate')
	parser.add_argument('--max_epochs', metavar='', type=int, help='maximum number of epochs for training')
	parser.add_argument('--layer_type', metavar='', help='type of layer in GNN (GCN, GIN, GAT, etc.)')
	parser.add_argument('--display', metavar='', type=bool, help='toggle display messages showing training progress')
	parser.add_argument('--device', metavar='', type=str, help='name of CUDA device to use or CPU')
	parser.add_argument('--eval_every', metavar='X', type=int, help='calculate validation/test accuracy every X epochs')
	parser.add_argument('--stopping_criterion', metavar='', type=str, help='model stops training when this criterion stops improving (can be train, validation, or test)')
	parser.add_argument('--stopping_threshold', metavar='T', type=float, help="model perceives no improvement when it does worse than (best loss) * T")
	parser.add_argument('--patience', metavar='P', type=int, help='model stops training after P epochs with no improvement')
	parser.add_argument('--train_fraction', metavar='', type=float, help='fraction of the dataset to be used for training')
	parser.add_argument('--validation_fraction', metavar='', type=float, help='fraction of the dataset to be used for validation')
	parser.add_argument('--test_fraction', metavar='', type=float, help='fraction of the dataset to be used for testing')
	parser.add_argument('--dropout', metavar='', type=float, help='layer dropout probability')
	parser.add_argument('--weight_decay', metavar='', type=float, help='weight decay added to loss function')
	parser.add_argument('--hidden_dim', metavar='', type=int, help='width of hidden layer')
	parser.add_argument('--hidden_layers', metavar='', type=ast.literal_eval, help='list containing dimensions of all hidden layers')
	parser.add_argument('--num_layers', metavar='', type=int, help='number of hidden layers')
	parser.add_argument('--batch_size', metavar='', type=int, help='number of samples in each training batch')


	arg_values = parser.parse_args()
	return arg_values