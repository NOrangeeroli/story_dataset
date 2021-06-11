name = 'test.txt'

gen = [x.strip() for x in open('gen_'+name,'r').readlines()]
ref = [x.strip() for x in open(name,'r').readlines()]

gen = [x.replace(' ','') for x in gen]
ref = [x.replace(' ','') for x in ref]
gen = [x.split('[S]')[1].split('<|endoftext|>')[0] for x in gen]
ref = [x.split('[S]')[1] for x in ref]

result = [True if x == y else False for x ,y in zip(gen,ref)]

import pdb;pdb.set_trace()
