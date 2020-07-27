import csv
import itertools
import sys
import pandas as pd
import numpy as np

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}

def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data

def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


people = load_data('data/family0.csv')
#names = set(people)

one_gene = {'Harry'}
two_genes = {'James'}
have_trait = {'James'}

#def joint_probability(people, one_gene, two_genes, have_trait):

names = [name for name in set(people)]

genes = []
for name in set(people):
    if name in one_gene:
        genes.append(1)
    elif name in two_genes:
        genes.append(2)
    else:
        genes.append(0)

mother_genes = []
for name in set(people):
    mother = people[name]['mother']
    if mother == None:
        mother_genes.append(np.nan)
    elif mother in one_gene:
        mother_genes.append(1)
    elif mother in two_genes:
        mother_genes.append(2)
    else:
        mother_genes.append(0)

father_genes = []
for name in set(people):
    father = people[name]['father']
    if father == None:
        father_genes.append(np.nan)
    elif father in one_gene:
        father_genes.append(1)
    elif father in two_genes:
        father_genes.append(2)
    else:
        father_genes.append(0)

trait = []
for name in set(people):
    if name in have_trait:
        trait.append(True)
    else: trait.append(False)



listofkeys = ('name', 'genes', 'mother_genes', 'father_genes', 'trait')
listofvalues = (names, genes, mother_genes, father_genes, trait)
dictionary=dict(zip(listofkeys,listofvalues))
df = pd.DataFrame(dictionary)
df.set_index('name', inplace=True)

def scenarios(row):
    # No parent has gene
    if (row['mother_genes'] == 0) and row['father_genes'] == 0:
        return 'p_00'
    # Both parents have one gene
    if (row['mother_genes'] == 1) and row['father_genes'] == 1:
        return 'p_11'
    # Both parents have two genes
    if (row['mother_genes'] == 2) and row['father_genes'] == 2:
        return 'p_22'
    # One parent has zero genes, one parent has one gene 
    if (row['mother_genes'] == 0) and row['father_genes'] == 1:
        return 'p_01'
    if (row['mother_genes'] == 1) and row['father_genes'] == 0:
        return 'p_01'
    # One parent has zero genes, one parent has two genes 
    if (row['mother_genes'] == 0) and row['father_genes'] == 2:
        return 'p_02'
    if (row['mother_genes'] == 2) and row['father_genes'] == 0:
        return 'p_02'
    # One parent has one gene, one parent has two genes 
    if (row['mother_genes'] == 1) and row['father_genes'] == 2:
        return 'p_12'
    if (row['mother_genes'] == 2) and row['father_genes'] == 1:
        return 'p_12'
    
# Scenarios:
df['scenario'] = df.apply(scenarios, axis=1)
#print(df)

def con_prob_0(row):
    # Probability of getting no gene dependent on parents genes
    if row['scenario'] == 'p_00':
        return 0.99 * 0.99
    if row['scenario'] == 'p_11':
        return 0.5 * 0.5
    if row['scenario'] == 'p_22':
        return 0.01 * 0.01
    if row['scenario'] == 'p_01':
        return 0.50 * 0.99
    if row['scenario'] == 'p_02':
        return 0.99 * 0.01
    if row['scenario'] == 'p_12':
        return 0.50 * 0.01

def con_prob_1(row):
    # Probability of getting one gene dependent on parents genes
    if row['scenario'] == 'p_00':
        return 0.01 * 0.99 + 0.01 * 0.99
    if row['scenario'] == 'p_11':
        return 0.50 * 0.50 + 0.50 * 0.50
    if row['scenario'] == 'p_22':
        return 0.99 * 0.01 + 0.01 * 0.99
    if row['scenario'] == 'p_01':
        return 0.50 * 0.99 + 0.01 * 0.50
    if row['scenario'] == 'p_02':
        return 0.99 * 0.99 + 0.01 * 0.01
    if row['scenario'] == 'p_12':
        return 0.99 * 0.50 + 0.01 * 0.50

def con_prob_2(row):
    # Probability of getting two genes dependent on parents genes
    if row['scenario'] == 'p_00':
        return 0.01 * 0.01
    if row['scenario'] == 'p_11':
        return 0.5 * 0.5
    if row['scenario'] == 'p_22':
        return 0.99 * 0.99
    if row['scenario'] == 'p_01':
        return 0.50 * 0.01
    if row['scenario'] == 'p_02':
        return 0.99 * 0.01
    if row['scenario'] == 'p_12':
        return 0.50 * 0.99

df['con_prob_0'] = df.apply(con_prob_0, axis=1)
df['con_prob_1'] = df.apply(con_prob_1, axis=1)
df['con_prob_2'] = df.apply(con_prob_2, axis=1)

#df['un_prob_gene'] = df.apply (lambda row: PROBS['gene'][row.genes], axis=1)

def prob_gene(row):
    if row['scenario'] == None:
        return PROBS['gene'][row.genes]
    else:
        if row.name in one_gene:
            return con_prob_1(row)
        elif row.name in two_genes:
            return con_prob_2(row)
        else:
            return con_prob_0(row)
                
df['prob_gene'] = df.apply(prob_gene, axis=1)



#df['un_prob_trait'] = df.apply (lambda row: PROBS['trait'][row.genes][row.trait], axis=1)
# Combined probability of genes and trait is zero if we know the parent
#df['un_prob_combined'] = df.apply (lambda row: row.un_prob_gene * row.un_prob_trait * (1 - row.parents), axis=1)

#df.loc[df['genes'] == 1, 'con_prob_gene'] = df.apply (lambda row: row.parents + 1, axis=1)
print(df)


