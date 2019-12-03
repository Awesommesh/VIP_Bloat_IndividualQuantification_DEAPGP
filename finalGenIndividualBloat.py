#!/usr/bin/env python
import ast
import re
import inspect
import argparse as ap
import xml.etree.ElementTree as ET
from lxml import etree
from deap import gp, base
from deap import creator
import itertools
from GPFramework.gp_framework_helper import LearnerType
import GPFramework.launchEMADE
from GPFramework.launchEMADE import objective_params, statistics_params, evoluation_params, misc_params
import os
from deap.gp import Primitive, PrimitiveTree, graph
import numpy as np
import pandas as pd
from GPFramework.data import EmadeDataPair
from GPFramework.EMADE import my_str
import GPFramework.gp_framework_helper as gpFrameworkHelper
from GPFramework.gp_framework_helper import EnsembleType
from GPFramework import EMADE as emade

def dataset_params(tree) -> dict:
    dataset_dict = {}
    datasetList = tree.iter('dataset')
    for datasetNum, dataset in enumerate(datasetList):
        # Iterate over each dataset and add to dictionary
        monte_carlo = dataset.iter('trial')
        dataset_dict[datasetNum] = {'name': dataset.findtext('name'),
                                   'type': dataset.findtext('type'),
                                   'batchSize': None if dataset.findtext('batchSize') is None else int(dataset.findtext('batchSize')),
                                   'trainFilenames':[],
                                   'testFilenames':[]}
        # The data is already folded for K-fold cross validation. Add these to the trainFilenames
        # and testFilenames lists
        for trial in monte_carlo:
            dataset_dict[datasetNum]['trainFilenames'].append(
                trial.findtext('trainFilename'))
            dataset_dict[datasetNum]['testFilenames'].append(
                trial.findtext('testFilename'))
    return dataset_dict

# Takes an XML file to get dataset, objective, and evaluation information
parser = ap.ArgumentParser()
parser.add_argument(
    'filename', help='Input to EMADE, see inputSample.xml')
parser.add_argument('-w', '--worker', dest='worker', default=False, action='store_true', help='Only run workers')

args = parser.parse_args()
inputFile = args.filename


# Valid XML file with inputSchema.xsd using lxml.etree
schema_doc = etree.parse(os.path.join('templates', 'inputSchema.xsd'))
schema = etree.XMLSchema(schema_doc)

doc = etree.parse(inputFile)
# Raise error if invalid XML
try:
    schema.assertValid(doc)
except:
    raise

# Uses xml.etree.ElementTree to parse the XML
tree = ET.parse(inputFile)
root = tree.getroot()

def str2bool(v):
  return v is not None and v.lower() in ("yes", "true", "t", "1")

def cache_params(tree) -> dict:
  # Initializes the cache dictionary
    cacheInfo = root.find('cacheConfig')
    # Get database information
    db_info = root.find('dbConfig')
    server = db_info.findtext('server')
    username = db_info.findtext('username')
    password = db_info.findtext('password')
    database = db_info.findtext('database')
    reuse = int(db_info.findtext('reuse'))
    database_str = 'mysql://' + username + ':' + password + '@' + server + '/' + database
    return {'cacheLimit': float(cacheInfo.findtext('cacheLimit')),
            'central': str2bool(cacheInfo.findtext('central')),
            'compression': str2bool(cacheInfo.findtext('compression')),
            'useCache': str2bool(cacheInfo.findtext('useCache')),
            'timeThreshold': float(cacheInfo.findtext('timeThreshold')),
            'timeout': cacheInfo.findtext('timeout'),
            'masterWaitTime': int(cacheInfo.findtext('masterWaitTime')),
            'database': database_str}

# Get python information
python_info = root.find('pythonConfig')
grid_python_command = python_info.findtext('gridPythonPath')
local_python_command = python_info.findtext('localPythonCommand')
slurm_worker_python_command = python_info.findtext('slurmPythonPathWorker')
slurm_master_python_command = python_info.findtext('slurmPythonPathMaster')

# Initializes the dataset dictionary
datasetDict = dataset_params(root)
print('Dataset dict')
print(datasetDict)
print()

# Initializes the objective dictionary
objectiveDict = objective_params(root)
print('Objectives dict')
print(objectiveDict)
print()

# Initializes the statistics dictionary
statisticsDict = statistics_params(root.find('statistics'))
print('Stats dict')
print(statisticsDict)
print()

# Initializes the evolution parameters dictionary
evolution_dict = evoluation_params(root.find('evolutionParameters'))
print('Evolution dict')
print(evolution_dict)
print()

# Initializes the miscellanious dictionary
misc_dict = misc_params(root)
print('Misc dict')
print(misc_dict)
print()

# Initializes the cache parameters dictionary
cache_dict = cache_params(root)
wait_time = cache_dict['masterWaitTime']
print('Cache dict')
print(cache_dict)
print()

# Get database information
db_info = root.find('dbConfig')
server = db_info.findtext('server')
username = db_info.findtext('username')
password = db_info.findtext('password')
database = db_info.findtext('database')
reuse = int(db_info.findtext('reuse'))

regression = root.findtext('regression')
if regression is None:
    gpFrameworkHelper.set_regression(False)
    regression = 0
elif regression == 1:
    gpFrameworkHelper.set_regression(True)
    regression = 1
else:
    gpFrameworkHelper.set_regression(False)
    regression = 0

emade.create_representation(adfs=3, regression=regression)

emade.setObjectives(objectiveDict)
emade.setDatasets(datasetDict)
emade.setMemoryLimit(misc_dict['memoryLimit'])
emade.setCacheInfo(cache_dict)
emade.set_statistics(statisticsDict)
emade.buildClassifier()

# Initialize pset
pset = gp.PrimitiveSetTyped("MAIN", [EmadeDataPair], EmadeDataPair)
gpFrameworkHelper.addTerminals(pset)
gpFrameworkHelper.addPrimitives(pset)
terminals = {}
primitives = {}
ephemerals = {}
for item in pset.mapping:
    if isinstance(pset.mapping[item], gp.Terminal):
        terminals[item] = pset.mapping[item]
    elif isinstance(pset.mapping[item], gp.Primitive):
        primitives[item] = pset.mapping[item]
names = []
methods = dir(gp)
for method in methods:
	pointer = getattr(gp, method)
	if inspect.isclass(pointer) and issubclass(pointer, gp.Ephemeral):
		ephemerals[method] = pointer

weights = tuple([objectiveDict[objective]['weight'] for objective in objectiveDict])
goals = tuple([objectiveDict[objective]['goal'] for objective in objectiveDict])
achievable = tuple([objectiveDict[objective]['achievable'] for objective in objectiveDict])
LROI = np.array(goals)
creator.create("FitnessMin", base.Fitness, weights=weights)
fitness_names = (datasetDict[dataset]['name'] for dataset in datasetDict)
fitness_attr = dict(zip(fitness_names, itertools.repeat(creator.FitnessMin)))

creator.create("Individual", list, pset=pset, fitness=creator.FitnessMin, age=0, hash_val=None, **fitness_attr)


def parse_tree(line, mode=0):
    """
    Parses the string of a seed assuming the seed is a tree

    Args:
        line (string): string of the tree
        mode (int):  0 for a normal primitive, 1 for a learner, 2 for an ensemble

    Returns:
        List of deap.gp objects (Primitives, Terminals, and Ephemerals)
    """
    # setup variables
    my_func = []
    node = ""
    wait = False
    parse = True
    i = 0

    # parse string until end of ephemeral/primitive is reached
    while parse:

        if line[i] == ")" and node == "":
            return my_func, i

        if (line[i] == "(" or line[i] == ")" or line[i] == "," or line[i - 1] == "}" or line[
            i - 1] == "]") and not wait:
            # remove the space before the "," if it exists
            if node[-1] == " ":
                node = node[:-1]
            elif node[0] == " ":
                node = node[1:]

            # figure out what node is
            if node in primitives:
                # add primitive to my_func since we're done parsing it
                my_func.append(primitives[node])

                # recursive call given rest of un-parsed line
                their_func, x = parse_tree(line[i + 1:])

                # skip over ")" and "," after returning the recursive call
                # this prevents checking the next string too early
                while (i + x + 1 < len(line)) and (line[i + x + 1] == ")" or line[i + x + 1] == ","):
                    x += 1

                # update current list and index
                my_func += their_func
                i += x
            elif node in terminals:
                # add terminal to my_func since we're done parsing it
                # this is mainly used for ARG0
                my_func.append(terminals[node])
            elif node == "learnerType":
                # recursive call with mode set for learnerType
                their_func, x = parse_tree(line[i + 1:], mode=1)

                # skip over ")" and "," after returning the recursive call
                # this prevents checking the next string too early
                while (i + x + 1 < len(line)) and (line[i + x + 1] == ")" or line[i + x + 1] == ","):
                    x += 1

                # update current list and index
                my_func += their_func
                i += x
            elif node == "ensembleType":
                # recursive call with mode set for ensembleType
                their_func, x = parse_tree(line[i + 1:], mode=2)

                # skip over ")" and "," after returning the recursive call
                # this prevents checking the next string too early
                while (i + x + 1 < len(line)) and (line[i + x + 1] == ")" or line[i + x + 1] == ","):
                    x += 1

                # update current list and index
                my_func += their_func
                i += x
            elif node in ephemeral_methods:
                # handles ephemerals using method references
                ephem = ephemerals[ephemeral_methods[node]]()
                ephem.value = pset.context[node]
                my_func.append(ephem)
            else:
                # assume the node is a literal (float, int, string, list, dict, etc.)
                node = ast.literal_eval(node)
                if isinstance(node, int):
                    if node <= 15:
                        ephem = ephemerals['myRandInt']()
                        ephem.value = node
                    elif node > 15 and node <= 100:
                        ephem = ephemerals['myMedRandInt']()
                        ephem.value = node
                    elif node > 100:
                        ephem = ephemerals['myBigRandInt']()
                        ephem.value = node
                elif isinstance(node, float):
                    ephem = ephemerals['myGenFloat']()
                    ephem.value = node
                elif isinstance(node, list):
                    ephem = ephemerals['listGen']()
                    ephem.value = node
                elif (isinstance(node, dict) or isinstance(node, str) or node is None) and mode > 0:
                    # in this case we just want the value
                    ephem = node
                else:
                    print("DEBUG (Node that was unsupported):", node)
                    raise ValueError("Unsupported type used as an argument.")

                my_func.append(ephem)

            # reset node to empty string
            # at this point we no longer need the old string because it's already processed
            node = ""

        else:
            # handle dicts and lists
            if line[i] == "{" or line[i] == "[":
                wait = True

            # add the char to node
            node += line[i]

        # check if end of ephemeral/primitive has been reached and if end of list or dict has been reached
        if line[i] == ")":
            parse = False
        elif line[i] == "}" or line[i] == "]":
            wait = False

        # increment index
        i += 1

    if mode == 1:
        # construct learnerType
        learner = ephemerals['learnerGen']()
        learner.value.name = my_func[0]
        learner.value.params = my_func[1]
        my_func = [learner]
    elif mode == 2:
        # construct ensembleType
        ensemble = ephemerals['ensembleGen']()
        ensemble.value.name = my_func[0]
        ensemble.value.params = my_func[1]
        my_func = [ensemble]

    return my_func, i

def parseInd(line):
    individual_list = []
    indicies = [line.start() for line in re.finditer('\(', line)]
    close_paren = [line.start() for line in re.finditer('\)', line)]
    colen = [line.start() for line in re.finditer(':', line)]
    open_bracket = [line.start() for line in re.finditer('{', line)]
    close_bracket = [line.start() for line in re.finditer('}', line)]

    indicies.extend(close_paren)
    indicies.extend(colen)
    indicies.extend(open_bracket)
    indicies.extend(close_bracket)
    indicies.sort()

    num = 0
    for index in indicies:
        if line[index + num] == '(':
            line = line[:index + num] + ',(,' + line[index + num + 1:]
        elif line[index + num] == ')':
            line = line[:index + num] + ',),' + line[index + num + 1:]
        elif line[index + num] == ':':
            line = line[:index + num] + ',:,' + line[index + num + 1:]
        elif line[index + num] == '{':
            line = line[:index + num] + ',{,' + line[index + num + 1:]
        elif line[index + num] == '}':
            line = line[:index + num] + ',},' + line[index + num + 1:]
        num += 2
    line = line.replace('\'', '')
    line = line.replace(',,', ',')
    line_list = line.split(',')
    new_line_list = []
    for item in line_list:
        item = item.strip()
        new_line_list.append(item)
    line_list = new_line_list
    line_list = [x for x in line_list if x != '' and x != '(' and x != ')' and x != ':']
    individual_list.append(line_list)
    return individual_list

# Calls learnerGen in gp_framework_helper.py to generate a LearnerType object
def get_machine_learner(learnerName, learnerParams, ensemble, ensembleParams):
    learner = ephemerals['learnerGen']()
    learner.value.learnerName = learnerName
    learner.value.learnerParams = learnerParams
    learner.value.ensemble = ensemble
    learner.value.ensembleParams = ensembleParams
    return learner

def runInd(individual_list):
    pset.context['learnerType'] = LearnerType
    for individual in individual_list:
        print("Running Individual")
        my_func = []
        machine_learner_params = {}
        gen_list = []
        learner_name_bool = False
        learner_params_bool = False
        learner_params_none = False
        first_learner_name = True
        first_learner_params = True
        param_name_bool = True
        gen_list_bool = False
        append = True

        for node in individual:
            if (node == 'ModifyLearnerEnsembleFloat'):
                node = 'ModifyLearnerFloat'
            if (node == 'ModifyLearnerEnsembleInt'):
                node = 'ModifyLearnerInt'
            if (node == 'passTriState'):
                continue
            if (node == 'passQuadState'):
                continue
            if (node == 'passBool'):
                continue
            if (node == 'passFloat'):
                continue
            if (node == 'passList'):
                continue
            if (node == 'passInt'):
                continue
            if (node == 'adf_1'):
                continue
            #print("Running")
            if node == 'learnerType':
                learner_name_bool = True
                append = False
            elif node == 'None':
                learner_params_none = True
                append = False
            elif node == '{':
                learner_params_bool = True
                append = False
            elif node == '}':
                first_learner_name = True
                first_learner_params = True
                learner_params_bool = False
                my_learner = get_machine_learner(learner_name, machine_learner_params, 'SINGLE', None)
                my_func.append(my_learner)
                append = False
                break
            elif '[' in node:
                gen_list_bool = True
                node = node.replace('[', '')
                append = False
            elif ']' in node:
                gen_list_bool = False
                node = node.replace(']', '')
                append = False
                gen_list.append(int(node))
                my_gen_list = ephemerals['listGen']()
                my_gen_list.value = gen_list
                my_func.append(my_gen_list)

            if gen_list_bool:
                gen_list.append(int(node))
                append = False

            if learner_name_bool:
                if first_learner_name:
                    first_learner_name = False
                else:
                    learner_name = node
                    learner_name_bool = False
                append = False

            if learner_params_bool:
                if first_learner_params:
                    first_learner_params = False
                elif param_name_bool:
                    param_name = node
                    param_name_bool = False
                else:
                    if '.' in node:
                        param_value = float(node)
                    elif type(node) == "boolean":
                        param_value = bool(node)
                    else:
                        param_value = int(node)
                    param_name_bool = True
                    machine_learner_params[param_name] = param_value
                append = False

            if learner_params_none:
                my_learner = get_machine_learner(learner_name, None, 'SINGLE', None)
                my_func.append(my_learner)
                learner_params_none = False
                append = False

            if append:
                if node in primitives:
                    my_func.append(primitives[node])
                elif node == 'ARG0' or node == 'trueBool' or node == 'falseBool':
                    my_func.append(terminals[node])
                elif '.' in node:
                    my_float = ephemerals['myGenFloat']()
                    my_float.value = float(node)
                    my_func.append(my_float)
                elif node == 'Tri0' or node == 'Tri1' or node == 'Tri2':
                    my_int = ephemerals['myRandTri']()
                    my_int.value = int(node[-1])
                    my_func.append(my_int)
                elif node == 'Quad0' or node == 'Quad1' or node == 'Quad2' or node == 'Quad3':
                    my_int = ephemerals['myRandQuad']()
                    my_int.value = int(node[-1])
                    my_func.append(my_int)
                elif node == "ADABOOST":
                    ensemble = ephemerals['ensembleGen']()
                    ensemble.value.name = node
                    ensemble.value.params = {'n_estimators': 50, 'learning_rate':1.0}
                    my_func.append(ensemble)
                elif node == "SINGLE":
                    ensemble = ephemerals['ensembleGen']()
                    ensemble.value.name = node
                    ensemble.value.params = None
                    my_func.append(ensemble)
                elif node == "BAGGED":
                    ensemble = ephemerals['ensembleGen']()
                    ensemble.value.name = node
                    ensemble.value.params = None
                    my_func.append(ensemble)
                elif node == "GRID":
                    ensemble = ephemerals['ensembleGen']()
                    ensemble.value.name = node
                    ensemble.value.params = None
                    my_func.append(ensemble)
                else:
                    if int(node) <= 15:
                        my_int = ephemerals['myRandInt']()
                        my_int.value = int(node)
                        my_func.append(my_int)
                    elif int(node) > 15 and int(node) <= 100:
                        my_int = ephemerals['myMedRandInt']()
                        my_int.value = int(node)
                        my_func.append(my_int)
                    elif int(node) > 100:
                        my_int = ephemerals['myBigRandInt']()
                        my_int.value = int(node)
                        my_func.append(my_int)

            append = True
        my_tree = gp.PrimitiveTree(my_func)
        my_individual = creator.Individual([my_tree, my_tree, my_tree, my_tree])

        # print the seeded individual to there terminal
        print(my_str(my_individual))

        # print the seeded individual to there terminal
        print(str(my_tree))
        return my_tree
#print (pset.primitives)
#tree = gp.PrimitiveTree.from_string("SpectralClustering(MeanWithHole(ARG0, 0, 0, 3, 32), greaterThan(0.1, 0.01), 10.0)", pset)
#Get all final generation individuals from database
'''test, test1 = parse_tree("SpectralClustering(MeanWithHole(ARG0, 0, 0, 3, 32), greaterThan(0.1, 0.01), 10.0)",  0)
tree = gp.PrimitiveTree(test)
print(test)
print(str(tree))'''
allIndv = pd.read_csv("/Users/animeshagrawal/eric_fork/emade/src/GPFramework/ind2.csv",usecols=[5,7])
allFinalGenTree = allIndv.loc[allIndv["evaluation_gen"]==max(allIndv["evaluation_gen"])]['tree']
'''index = 66
print(allFinalGenTree[allFinalGenTree.index[index]])
print(parseInd(allFinalGenTree[allFinalGenTree.index[index]]))
print(runInd(parseInd(allFinalGenTree[allFinalGenTree.index[index]])))
test, test1 = parse_tree(allFinalGenTree[allFinalGenTree.index[index]],  0)
tree = gp.PrimitiveTree(test)
print(test)
print(str(tree))'''

deapTrees = []

count=0
success=0
for indv in allFinalGenTree:
    #try:'''
    '''
    test, test1 = parse_tree(indv, 0)
    tree = gp.PrimitiveTree(test)'''
    print(indv)
    tree = runInd(parseInd(indv))
    if (str(tree) != indv):
        count+=1
        print("Original: " + indv)
        print("Converted: " + str(tree))
    else:
        deapTrees.append(tree)
        success+=1
print(success)
print(count)
print(allFinalGenTree.index.size)

#work
def getSubtreeSlices(ind):
    subtrees = []
    for i in range(len(ind)):
        subtrees.append(ind.searchSubtree(i))
    return subtrees


def subtreeGenerator(subtreeSlices, treeBestInd):
    listOfPrimSubTF = []
    for j in range(len(subtreeSlices)):
        primSubTreeL = []
        for i in range(subtreeSlices[j].start, subtreeSlices[j].stop):
            primSubTreeL.append(treeBestInd[i])
        primSubTree = gp.PrimitiveTree(primSubTreeL)
        toolbox = base.Toolbox()
        toolbox.register("compile", gp.compile, pset=pset)
        listOfPrimSubTF.append(toolbox.compile(expr=primSubTree))
    return listOfPrimSubTF


def indvBloatIdentifier(indSubtreeFs, slices):
    sols = []
    for i in range(len(indSubtreeFs)):
        if i == 0:
            print("HERE")
        emade._inst.pset.context['learnerType'] = LearnerType
        data_pair = emade._inst.datasetDict[0]['dataPairArray'][0]
        train = data_pair.get_train_data().get_instances()
        test = data_pair.get_test_data().get_instances()
        truth_data = emade._inst.datasetDict[0]['truthDataArray'][0]
        #print(type(data_pair), type(truth_data))
        #test = indSubtreeFs[i](data_pair)
        '''try:
            result = indSubtreeFs[i](data_pair) # my train = train data
        except:
            continue'''
    #for i in range(len(sols)):
     #   print(type(sols[i]))
    '''isBloat = [None] * len(sols)
    alreadyBloat = set()
    for i in range(len(sols)):
        if i in alreadyBloat:
            continue
        aVal = None
        notConst = True
        for j in range(len(sols[i])):
            if aVal == None:
                aVal = sols[i][j]
            else:
                if aVal != sols[i][j]:
                    notConst = True
                    break
        isBloat[i] = not notConst
        if isBloat[i]:
            j = i + 1
            while j < len(slices):
                if slices[j].start > slices[i].start and slices[j].stop <= slices[i].stop:
                    alreadyBloat.add(j)
                    isBloat[j] = True
                j += 1
        nodeRootChildren = set()  # Check if just root node is bloat
        startRange = slices[i].start + 1
        j = i + 1
        while j < len(slices) and startRange != slices[i].stop:
            if slices[j].start == startRange:
                startRange = slices[j].stop
                nodeRootChildren.add(j)
            j += 1
        for child in nodeRootChildren:
            if np.array_equal(sols[i], sols[child]):
                isBloat[i] = True
    numNonBloat = 0
    for i in range(len(isBloat)):
        if not isBloat[i]:
            numNonBloat += 1
    return (numNonBloat / len(isBloat)) * 100, isBloat'''
    return None, None


def clean(individual, bloatNodes):
    return individual, bloatNodes  # to be completed

#run evaluate bloat and clean on all individuals in final gen

for deapTree in deapTrees:
    print(deapTree)
    subtrees = getSubtreeSlices(deapTree)
    listOfPrimSubTF = subtreeGenerator(subtrees, deapTree)
    score, bloatArr = indvBloatIdentifier(listOfPrimSubTF, subtrees)
    print("Score:", score, "percent")
    print(" ")