from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
from primitives import standard_env, Symbol, Env
import torch
from sampler import sampler

def evaluate_program(ast):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """
    for i in range(len(ast)):
        ei = eval(ast[i])
        if ei != None:
            res = ei
    return res, None

class Procedure(object):
    "A user-defined Scheme procedure."
    def __init__(self, parms, body, env):
        self.parms, self.body, self.env = parms, body, env
    def __call__(self, *args): 
        return eval(self.body, Env(self.parms, args, self.env))


def eval(x, env=standard_env()):
    "Evaluate an expression in an environment."
    if isinstance(x, Symbol):    # variable reference
        return env.find(x)[x]
    elif not isinstance(x, list): # constant 
        return torch.tensor(x)
    op, *args = x       
    if op == 'if':             # conditional
        (test, conseq, alt) = args
        exp = (conseq if eval(test, env) else alt)
        return eval(exp, env)
    elif op == 'defn':         # definition
        (string, parms, body) = args
        env[string] = Procedure(parms, body, env)
        return None
    elif op == 'let':
        (symbol, exp) = args[0]
        env[symbol] = eval(exp, env)
        return eval(args[1], env)
    elif op == 'sample':
        dist = eval(args[0],env)
        return dist.sample()
    elif op == 'observe':
        dist = eval(args[0],env)
        return dist.sample() 
    else:                        # procedure call
        proc = eval(op, env)
        vals = [x for x in (eval(arg, env) for arg in args)]
        return proc(*vals)


def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast)
    
def run_deterministic_tests():
    
    for i in range(1,14):
        #note: this path should be with respect to the daphne path!
        ast = daphne(['desugar', '-i', 'C:/Users/jlovr/CS532-HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('C:/Users/jlovr/CS532-HW2/programs/tests/deterministic/test_{}.truth'.format(i))
        ret, sig = evaluate_program(ast)
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,ast))
        
        print(i, 'Test passed, returned', ret)
        
    print('All deterministic tests passed')
    

def run_probabilistic_tests():
    
    num_samples= 1e4
    max_p_value = 1e-4
    
    for i in range(1,7):
        #note: this path should be with respect to the daphne path!        
        ast = daphne(['desugar', '-i', 'C:/Users/jlovr/CS532-HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('C:/Users/jlovr/CS532-HW2/programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(ast)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        assert(p_val > max_p_value)

        print('Test {} passed,'.format(i), 'p value', p_val)

    
    print('All probabilistic tests passed')    

        
if __name__ == '__main__':
    # print("deterministic tests \n")
    # run_deterministic_tests()
    # print("\n\n\n")
    # print("probabilistic tests \n")
    # run_probabilistic_tests()
    # print("\n\n\n")

    n = 1000

    for i in range(1,5):
        ast = daphne(['desugar', '-i', 'C:/Users/jlovr/CS532-HW2/programs/{}.daphne'.format(i)])
        samples = []
        for _ in range(n):
            samples.append(evaluate_program(ast)[0])
        sampler(i, samples, "_standard.pdf")

        # print(ast, "\n")
        # print('Sample of prior of program {}:'.format(i))
        # print(evaluate_program(ast)[0], "\n\n\n")



