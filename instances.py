import json
import argparse
import pprint, pickle

if __name__=='__main__':
	pp = pprint.PrettyPrinter(indent=1, width=120)

	inst_log = 'ec2-instances.json'

	with open(inst_log) as file:
		data = json.load(file)
		pp.pprint(data)