import argparse


def get_parser():
	parser = argparse.ArgumentParser(description='Experiment')

	parser.add_argument('-mode', type=str, required=True, help='Required.')
	parser.add_argument('-cfg_name', type=str, required=True, help='Required.')
	parser.add_argument('-v', type = bool, default = False)

	return parser
