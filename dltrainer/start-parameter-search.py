import yaml
import argparse

def gen_script(base_script, odir):

	with open(base_script) as f:
		config = yaml.load(f)
	
	lr_set = [0.00001, 0.0001, 0.001, 0.01, 0.1]
	decay_set = [1e-3, 1e-4, 1e-5, 1e-6]

	lr_key = '--lr'
	decay_key = '--weight-decay'
	temp = config['spec']['template']['spec']['containers'][0]['args']
	for lr in lr_set:
		for decay in decay_set:
			temp.append(lr_key)
			temp.append(lr)
			temp.append(decay_key)
			temp.append(decay)
			temp = ["{}str(i){}".format('\'', '\'') for i in temp]
			#temp = map(str, temp)
			with open('{}/deploy-lr-{}-decay-{}.yml'.format(odir,lr, decay), "w") as ftemp:
				yaml.dump(config, ftemp)
			
			temp = temp[:len(temp)-4]


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Generate kube scripts for parameter search')
	parser.add_argument('-b', help='Base trainer deployment script', required=True, type=str)
	parser.add_argument('-o', help='Base trainer deployment script', required=True, type=str)
	args = parser.parse_args()

	gen_script(args.b, args.o)
