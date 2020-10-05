import pynupack as nu # renamed the nupack directory from multistrand, slightly modified, used elsewhere
import random
import numpy as np
from Bio.Seq import Seq
import matplotlib.pyplot as plt
from Bio.SeqUtils import MeltingTemp as mt
# from Bio import AlignIO
# from Bio.Align import MultipleSeqAlignment
# from Bio import AlignIO
# from Bio import pairwise2
# from Bio.SeqRecord import SeqRecord
# from Bio.Align.Applications import MuscleCommandline
# from Bio.Alphabet import IUPAC
# from Bio.Align import AlignInfo
# from Bio.pairwise2 import format_alignment
# from Bio import SeqIO
# import Levenshtein

def get_base_sampling_key(pA = 0.3,pT = 0.3,pC = 0.2,pG = 0.2):
    base_probabilities = {"A":pA, "T":pT, "C":pC, "G":pG}
    pdistGTC =np.array([base_probabilities["G"],base_probabilities["T"],base_probabilities["C"]])/np.sum([base_probabilities["G"],base_probabilities["T"],base_probabilities["C"]])
    pdistATC =np.array([base_probabilities["A"],base_probabilities["T"],base_probabilities["C"]])/np.sum([base_probabilities["A"],base_probabilities["T"],base_probabilities["C"]])
    pdistAGC =np.array([base_probabilities["A"],base_probabilities["G"],base_probabilities["C"]])/np.sum([base_probabilities["A"],base_probabilities["G"],base_probabilities["C"]])
    pdistAGT =np.array([base_probabilities["A"],base_probabilities["G"],base_probabilities["T"]])/np.sum([base_probabilities["A"],base_probabilities["G"],base_probabilities["T"]])
    base_sampling_key = { "A":[["G","T","C"],pdistGTC],"G":[["A","T","C"],pdistATC],
                                "T":[["A","G","C"],pdistAGC],"C":[["A","G","T"],pdistAGT]}
    return base_sampling_key

def longest(s):
    maximum = count = 0
    current = ''
    for c in s:
        if c == current:
            count += 1
        else:
            count = 1
            current = c
        maximum = max(count,maximum)
    return maximum

def random_seq_of_length(length):
    seq = ''.join(random.choice("ATGC") for _ in range(length))
    return seq

def mutate(sequence, base_sampling_key):
    base = random.randrange(len(sequence))    
    new_base = np.random.choice(base_sampling_key[sequence[base]][0],p=base_sampling_key[sequence[base]][1])
    new_sequence = sequence[0:base] + new_base + sequence[base+1:]
#     print(new_sequence)
#     print(sequence)
    return new_sequence

def get_most_represented_base_freq(sequence):
    length1 = int(len(sequence)/2)
    freqA1 = float(sequence[0:length1].count("A"))/length1
    freqG1 = float(sequence[0:length1].count("G"))/length1
    freqC1 = float(sequence[0:length1].count("C"))/length1
    freqT1 = float(sequence[0:length1].count("T"))/length1
    max_freq1 = np.max([freqA1, freqG1, freqC1, freqT1])
    
    freqA2 = float(sequence[length1:].count("A"))/length1
    freqG2 = float(sequence[length1:].count("G"))/length1
    freqC2 = float(sequence[length1:].count("C"))/length1
    freqT2 = float(sequence[length1:].count("T"))/length1
    max_freq2 = np.max([freqA2, freqG2, freqC2, freqT2])    
    penalty = (1+(max_freq1-0.25))*1+(max_freq2-0.25)
    return penalty

# def tm():
# for seq in inp.values():
#     N = len(seq)
#     dA = seq.count("A")
#     dG = seq.count("G")
#     dC = seq.count("C")
#     dT = seq.count("T")
#     GC = float(dG+dC)/float(N) #in percent %
#     print(GC)
#     cation = sodium + 120*np.sqrt(magnesium - dNTPs)
#     owen_Tm = 87.16 + 0.345*GC + np.log(cation)*(20.17 - 0.066*GC) - .75*percentDMSO
    
#     print(owen_Tm)

def prime(sequence):
    complement = Seq(sequence).reverse_complement().__str__()
    return complement

def get_complements_list(domains):
    comp_list = []
    for dom in domains:
        comp_list += [Seq(dom).reverse_complement().__str__()]
    return comp_list

def get_interaction_score(domains, domains_c,T=25,magnesium=0, sodium=1.0):
    e_mat_domains = get_energy_matrix(domains, domains_c, T = T, magnesium = magnesium, sodium=sodium)
    target_duplex_energies = np.diag(e_mat_domains)
    cross_reactions = e_mat_domains-np.eye(len(domains))*target_duplex_energies
    problematic_row = np.argmax(np.sum(-1*cross_reactions,axis=0),axis=0)
    cross_reaction_score = np.sum(cross_reactions)
    target_reaction_score = np.sum(target_duplex_energies)
    master_score = target_reaction_score-cross_reaction_score
    return master_score,problematic_row

def get_quick_interaction_score(domains, domains_c,row,T=25,magnesium=0, sodium=1.0, tm_target = 60):
    no_domains = len(domains)
    no_domains_c = len(domains_c)
    e_vec_domains = np.zeros((no_domains))
    for i in range(0,no_domains):
        e_vec_domains[i] = get_duplex_energy([domains[row], domains_c[i]], T = T, magnesium = magnesium, sodium=sodium)
    target_score = e_vec_domains[row]
    cross_reactions = np.sum(np.delete(e_vec_domains, row))
    tm_wallace = mt.Tm_Wallace(domains[row])
    tm_penalty = 1+np.abs(tm_target-tm_wallace)
    quick_score = target_score-cross_reactions*longest(domains[row])*get_most_represented_base_freq(domains[row])*tm_penalty
    return quick_score
    
def get_energy_matrix(domains, domains_c,T=25,magnesium=0, sodium=1.0):
    no_domains = len(domains)
    no_domains_c = len(domains_c)
    e_mat_domains = np.zeros((no_domains,no_domains_c))
    for i in range(0,no_domains):
        for j in range(0, no_domains_c):
            e_mat_domains[i,j] = get_duplex_energy([domains[i], domains_c[j]], T = T, magnesium = magnesium, sodium=sodium)    
    return e_mat_domains

def optimize_domains_quick(input_domains_dict,fixed_domains_dict, iterations, algorithm_reversion_probability = 0.1,plot_interval=20,file_prefix="", T = 25, magnesium = 0.0, sodium=1.0):
    base_sampling_key = get_base_sampling_key(pA = .25, pT = .25, pC =.25, pG = .25)
    input_domains = [seq for seq,temp in input_domains_dict.values()]
    fixed_domains = fixed_domains_dict.values()
    tms = [temp for seq,temp in input_domains_dict.values()]
    domains = input_domains+fixed_domains+get_complements_list(input_domains+fixed_domains)    
    scores = np.zeros((np.round(iterations/plot_interval)))
    random_starting_domain =random.choice(range(0,len(input_domains)))
    for iteration in range(0,iterations):
        
        
        number_of_mutations = 1
        for mutation in range(0,number_of_mutations):
            domain_pick = (random_starting_domain + iteration)%len(input_domains) #problematic_row #random.choice(range(0,len(input_domains)))
            initial_score = get_quick_interaction_score(domains, get_complements_list(domains), row=domain_pick, T = T, magnesium = magnesium, sodium=sodium, tm_target = tms[domain_pick])
            proposed_new_domain = mutate(domains[domain_pick],base_sampling_key)
            if domain_pick == 0:
                proposed_new_input_domains = [proposed_new_domain]+domains[1:len(input_domains)]    

            else:
                proposed_new_input_domains = domains[0:domain_pick]+[proposed_new_domain]+domains[domain_pick+1:len(input_domains)]
            proposed_new_domains = proposed_new_input_domains+fixed_domains+get_complements_list(proposed_new_input_domains+fixed_domains)
            proposed_score = get_quick_interaction_score(proposed_new_domains, get_complements_list(proposed_new_domains),row=domain_pick, T = T, magnesium = magnesium, sodium=sodium, tm_target = tms[domain_pick])
        if proposed_score < initial_score or random.random() < algorithm_reversion_probability:
            
            domains = [] + proposed_new_domains
        print(iteration, initial_score, proposed_score)    
        if iteration%plot_interval == 0 or iteration == 0:
            print("new variable domains: ", domains[0:len(input_domains)])
            plot_energy_matrix(domains,name= file_prefix+"energy_matrix"+str(int(iteration/plot_interval))+".svg" )
#             print(iteration, scores)
#             scores[int(iteration/plot_interval)],problematic_row = get_interaction_score(domains, get_complements_list(domains), T = T, magnesium = magnesium, sodium=sodium)
    new_variable_domain_dict = dict(zip(input_domains_dict.keys(), zip(domains[0:len(input_domains)], tms)   ))
    return new_variable_domain_dict



def plot_energy_matrix(domains, name):
    no_domains = len(domains)
    no_domains_c = len(get_complements_list(domains))
    fig, ax1 = plt.subplots(1, 1,figsize=(20,20))
    ax1.set_yticks(np.arange(0,no_domains)+.5)

    ax1.imshow(get_energy_matrix(domains,get_complements_list(domains)),cmap="brg_r") 
    ax1.set_yticklabels([x for x in domains],rotation=0)
    ax1.set_xticks(np.arange(0,no_domains)+.5)
    ax1.set_xticklabels([x for x in get_complements_list(domains)],rotation=90)
    ax1.set_ylim(len(domains),0)
    ax1.set_xlim(len(domains),0)
    mesh = ax1.pcolormesh(get_energy_matrix(domains,get_complements_list(domains)),cmap="brg_r")
    cbar = fig.colorbar(mesh)
    cbar.set_label('$\Delta$ G', rotation=270)
    plt.savefig(name)
    plt.close()   
    
def get_duplex_energy(sequences, material = 'dna',ordering = None, dangles = 'some',
    T = 25,multi = False ,pseudo = False,sodium = 1.0,magnesium = 0.0 ,degenerate = False, full_output = False):
    #### compute annealed structures of hairpin strands ####
    args, cmd_input = \
    nu.setup_nupack_input(exec_name = 'mfe', sequences = sequences, ordering = ordering,
                       material = material, sodium = sodium, magnesium = magnesium,
                       dangles = dangles, T = T, multi = multi, pseudo = pseudo)
    output = nu.call_with_file(args, cmd_input, '.mfe')
    cleaned_output = list(enumerate(output))[0][1]
    try: 
        duplex_energy = float(cleaned_output[14].replace("\n", ""))
        if full_output == True:
            print(duplex_energy, cleaned_output[15].replace("\n", ""), sequences)
    except:
        duplex_energy = 0
    return duplex_energy