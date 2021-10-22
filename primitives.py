import torch
import operator as op

# rest, nth, conj, cons as described in the book

# Scheme type objects
Symbol = str              # A Scheme Symbol is implemented as a Python str
Number = (torch.int32, torch.float64, torch.float32)     # A Scheme Number is implemented as a Python int or float
Atom   = (Symbol, Number) # A Scheme Atom is a Symbol or Number
List   = torch.tensor             # A Scheme List is implemented as a Python list
Expression  = (Atom, List)     # A Scheme expression is an Atom or List

class Env(dict):
    "An environment: a dict of {'var': val} pairs, with an outer Env."
    def __init__(self, parms=(), args=(), outer=None):
        self.update(zip(parms, args))
        self.outer = outer
    def find(self, var):
        "Find the innermost Env where var appears."
        try: 
            return self if (var in self) else self.outer.find(var)
        except: 
            print("except", var)
            raise

class Normal(object):
    def __init__(self, *x):
        self.dist = torch.distributions.Normal(*torch.FloatTensor([*x]))

    def sample(self):
        return self.dist.sample()

    def log_prob(self, c):
        return self.dist.log_prob(c)

class Beta(object):
    def __init__(self, *x):
        self.dist = torch.distributions.Beta(*torch.FloatTensor([*x]))

    def sample(self):
        return self.dist.sample()

    def log_prob(self, c):
        return self.dist.log_prob(c)

class Gamma(object):
    def __init__(self, *x):
        self.dist = torch.distributions.Gamma(*torch.FloatTensor([*x]))

    def sample(self):
        return self.dist.sample()

    def log_prob(self, c):
        return self.dist.log_prob(c)

class Dirac(object):
    def __init__(self, x):
        self.value = x

    def sample(self):
        return self.value

    def log_prob(self, c):
        return c==self.value

class Bernoulli(object):
    def __init__(self, x):
        self.dist = torch.distributions.Bernoulli(x)
        self.p = x

    def sample(self):
        return self.dist.sample()

    def log_prob(self, c):
        if c:
            return torch.log(self.p)
        else:
            return torch.log(1 - self.p)

class Exponential(object):
    def __init__(self, *x):
        self.dist = torch.distributions.Exponential(*torch.FloatTensor([*x]))

    def sample(self):
        return self.dist.sample()

    def log_prob(self, c):
        return self.dist.log_prob(c)

class Uniform(object):
    def __init__(self, *x):
        self.dist = torch.distributions.Uniform(*torch.FloatTensor([*x]))

    def sample(self):
        return self.dist.sample()

    def log_prob(self, c):
        return self.dist.log_prob(c)

class Dirichlet(object):
    def __init__(self, x):
        self.dist = torch.distributions.Dirichlet(x)

    def sample(self):
        return self.dist.sample()

    def log_prob(self, c):
        return self.dist.log_prob(c)

class Discrete(object):
    def __init__(self, x):
        self.dist = torch.distributions.Categorical(probs = torch.FloatTensor(x))

    def sample(self):
        return self.dist.sample()

    def log_prob(self, c):
        return self.dist.log_prob(c)
        
def get(x,i):
    if type(x)==dict:
        return x[i.item()]
    else:
        return x[int(i)]

def put(x,i,v):
    if type(x)==dict:
        x[i.item()] = v
    else:
        x[int(i)] = v
    return x

def append(x, v):
    try:
        return torch.stack(list(x) + [v])
    except:
        return torch.stack([x,v])

def prepend(x, v):
    try:
        return torch.stack([v] + list(x))
    except:
        return torch.stack([v,x])

def hash_map(*x):
    return {x[i].item(): x[i + 1] for i in range(0, len(x), 2)}

def hash_map_graph(x):
    return {x[i].item(): x[i + 1] for i in range(0, len(x), 2)}

def vector(*x):
    try:
        return torch.stack([*x])
    except:
        return x
            
def last(x):
    try:
        return x[-1]
    except:
        return x

def first(x):
    try:
        return x[0]
    except:
        return x
    
def standard_env() -> Env:
    "An environment with some Scheme standard procedures."
    env = Env()
    env.update(vars(torch)) # sin, cos, sqrt, pi, ...
    env.update({
        '+':torch.add, '-':torch.sub, '*':torch.mul, '/':torch.div, 
        '>':torch.gt, '<':torch.lt, '>=':torch.ge, '<=':torch.le, '=':torch.eq, 
        'abs':     abs,
        'apply':   lambda proc, args: proc(*args),
        'begin':   lambda *x: x[-1],
        'car':     lambda x: x[0],
        'cdr':     lambda x: x[1:], 
        'cons':    lambda x,y: [x] + y,
        'eq?':     op.is_, 
        'expt':    pow,
        'equal?':  torch.eq, 
        'length':  len, 
        'list':    lambda *x: List(x), 
        'list?':   lambda x: isinstance(x, List), 
        'map':     map,
        'max':     max,
        'min':     min,
        'not':     op.not_,
        'empty?':   lambda x: x == [], 
        '=':        lambda x, y: x==y,
        'and':     lambda x, y: x and y,
        'or':     lambda x, y: x or y,
        'number?': lambda x: isinstance(x, Number),  
		'print':   print,
        'procedure?': callable,
        'round':   round,
        'symbol?': lambda x: isinstance(x, Symbol),
        'vector':  vector,
        'first':  first,
        'last':  last,
        'second': lambda x: x[1],
        'rest':  lambda x: x[1:],
        'get':  get,
        'put': put,
        'append': append,
        'hash-map': hash_map,
        'hash-map-graph': hash_map_graph,
        'normal': lambda *x: Normal(*x),
        'beta': lambda *x: Beta(*x),
        'exponential': lambda *x: Exponential(*x),
        'uniform': lambda *x: Uniform(*x),
        'discrete': lambda *x: Discrete(*x),
        'dirichlet': lambda *x: Dirichlet(*x),
        'flip': lambda *x: Bernoulli(*x),
        'gamma': lambda *x: Gamma(*x),
        'dirac': lambda *x: Dirac(*x),
        'mat-transpose': lambda x: x.T,
        'mat-tanh': lambda x: x.tanh(),
        'mat-mul': lambda x,y: torch.matmul(x.float(),y.float()),
        'mat-add': lambda x, y: x+y,
        'mat-repmat': lambda x,y,z: x.repeat((int(y), int(z)))
    })
    return env

 