import torch
from daphne import daphne

from primitives import standard_env 
from tests import is_tol, run_prob_test_graph,load_truth
from sampler import sampler

# Put all function mappings from the deterministic language environment to your
# Python evaluation context here:
env = standard_env()

def topological_helper(v,E,visited,stack):
    visited[v] = True

    if v in E:
        for u in E[v]:
            if visited[u] == False:
                topological_helper(u,E,visited,stack)
 
    stack.insert(0,v)

def topological(V, A):
    visited = dict.fromkeys(V,False)
    stack = []
    for v in V:
        if not visited[v]:
            topological_helper(v,A,visited,stack)
    return stack

def deterministic_eval(exp):
    "Evaluation function for the deterministic target language of the graph based representation."
    if type(exp) is list:
        op = exp[0]
        args = exp[1:]
        if op == 'if':             # conditional
            (test, conseq, alt) = args
            exp = (conseq if deterministic_eval(test) else alt)
            return deterministic_eval(exp)
        elif op == 'hash-map':
            return deterministic_eval(["hash-map-graph", torch.tensor([item for pair in args for item in pair])])
        else:
            return env[op](*map(deterministic_eval, args))
    elif type(exp) is int or type(exp) is float:
        # We use torch for all numerical objects in our evaluator
        return torch.tensor(float(exp))
    else: 
        return exp

def graph_eval(exp, values):
    if type(exp) == list:
        return [graph_eval(x, values) for x in exp]
    try:
        return values[exp] 
    except:
        return exp


def sample_from_joint(graph):
    "This function does ancestral sampling starting from the prior."
    values = {}
    # "V" contains all random variable vertices, 
    # "A" contains the forward pointing adjacency information, 
    # "P" contains the symbolic expressions of the link functions
    V, A, P = graph[1]['V'], graph[1]['A'], graph[1]['P']
    
    V_sorted = topological(V, A)

    for v in V_sorted:
        op, exp = P[v][0], P[v][1]
        if op == "sample*":
            link = graph_eval(exp, values)
            values[v] = deterministic_eval(link).sample()
        # ignore op == "observe*"

    # The last entry in the graph is the return value of the program.
    return deterministic_eval(graph_eval(graph[-1], values))


def get_stream(graph):
    """
    Return a stream of prior samples
    Args: 
        graph: json graph as loaded by daphne wrapper
    Returns: a python iterator with an infinite stream of samples
    """
    while True:
        yield sample_from_joint(graph)


def run_deterministic_tests():
    
    for i in range(1,14):
        #note: this path should be with respect to the daphne path!
        graph = daphne(['graph','-i','C:/Users/jlovr/CS532-HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = deterministic_eval(graph[-1])
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for graph {}'.format(ret,truth,graph))
        
        print(i, 'Test passed, returned', ret)
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    num_samples=1e4
    max_p_value = 1e-4
    
    for i in range(1,7):
        #note: this path should be with respect to the daphne path!        
        graph = daphne(['graph', '-i', 'C:/Users/jlovr/CS532-HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(graph)
        
        p_val = run_prob_test_graph(stream, truth, num_samples)
        
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
        graph = daphne(['graph','-i','C:/Users/jlovr/CS532-HW2/programs/{}.daphne'.format(i)])
        
        # samples = []
        # for _ in range(n):
        #     samples.append(sample_from_joint(graph))
        # sampler(i, samples, "_graph.pdf")

        print('\n\n\nSample of prior of program {}:'.format(i))
        print(sample_from_joint(graph))    

    