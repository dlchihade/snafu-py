from . import *

# given list of cluster lengths, compute average cluster size of each list, then return avearge of that
# also works on single list
def clusterSize(l, scheme, clustertype='fluid'):
    """
    Compute average cluster sizes from a list or list of lists.

    This function identifies clusters in the input data using a specified
    clustering scheme and type, then calculates the average cluster size 
    for each list. If a single list is passed, it still works and returns 
    a single averaged result.

    Parameters
    ----------
    l : list
        A list of cluster data or a list of lists, where each inner list 
        contains cluster lengths or indices.
    scheme : object
        Clustering scheme or method used to find clusters.
    clustertype : str, optional
        Type of clustering to apply. Default is 'fluid'.

    Returns
    -------
    list of float
        A list of average cluster sizes, one per input list (or a single 
        average if the input was a single list).
    """
    clist = findClusters(l, scheme, clustertype)
    
    avglists=[]
    for i in clist:
        avglist=[]
        for l in i:
            avglist.append(np.mean(l))
        avglists.append(np.mean(avglist))
    return avglists

# given list of cluster lengths, compute average number of cluster switches of each list, then return avearge of that
# also works on single list
def clusterSwitch(l, scheme, clustertype='fluid',switchrate=False):
    """
    Compute the number of cluster switches in a list or nested lists.

    Determines how often the category or type of cluster changes within 
    a list (or nested list) based on a clustering scheme. Optionally 
    returns the switch rate (normalized by list length).

    Parameters
    ----------
    l : list
        A list of items or a list of lists of items.
    scheme : object
        Clustering scheme (e.g., semantic or letter-based) used to define clusters.
    clustertype : str, optional
        Type of clustering logic to apply. Options are 'fluid' or 'static'. Default is 'fluid'.
    switchrate : bool, optional
        If True, returns the switch count as a proportion of list length. Default is False.

    Returns
    -------
    list of float
        The number or rate of cluster switches per list or sublist.
    """    
    clist = findClusters(l, scheme, clustertype)
    """
    Find and measure clusters in a list or list of lists.

    Clusters are formed based on overlapping categories (semantic or letter).
    Returns cluster sizes or nested lists of sizes depending on the input format.

    Parameters
    ----------
    l : list
        A list of items or a list of lists of items.
    scheme : object
        Clustering scheme to apply (e.g., a semantic mapping file or int for letter clusters).
    clustertype : str, optional
        Type of clustering logic. 'fluid' retains overlapping categories; 'static' requires consistency. 
        Default is 'fluid'.

    Returns
    -------
    list
        A list of cluster sizes or nested list of cluster sizes.
    """
    
    avglists=[]
    for inum, i in enumerate(clist):
        avgnum=[]
        if len(i) > 0:
            if isinstance(i[0], list):
                for lstnum, lst in enumerate(i):
                    switches = len(lst)-1
                    if switchrate:
                        switches = switches / len(l[inum][lstnum])
                    avgnum.append(switches)
                avglists.append(np.mean(avgnum))
            else:
                switches = len(i)-1
                if switchrate:
                    switches = switches / len(l[inum])
                avglists.append(switches)
        else:
            avglists.append(0)
    return avglists

# report average cluster size for list or nested lists
def findClusters(l, scheme, clustertype='fluid'):
    # only convert items to labels if list of items, not list of lists

    """
    Find and measure clusters in a list or list of lists.

    Clusters are formed based on overlapping categories (semantic or letter).
    Returns cluster sizes or nested lists of sizes depending on the input format.

    Parameters
    ----------
    l : list
        A list of items or a list of lists of items.
    scheme : object
        Clustering scheme to apply (e.g., a semantic mapping file or int for letter clusters).
    clustertype : str, optional
        Type of clustering logic. 'fluid' retains overlapping categories; 'static' requires consistency. 
        Default is 'fluid'.

    Returns
    -------
    list
        A list of cluster sizes or nested list of cluster sizes.
    """

    if len(l) > 0:
        if isinstance(l[0], list):
            clusters=l
        else:
            clusters=labelClusters(l, scheme)
    else:
        clusters=[]
    
    csize=[]
    curcats=set([])
    runlen=0
    clustList=[]
    firstitem=1
    for inum, item in enumerate(clusters):
        if isinstance(item, list):
            clustList.append(findClusters(item, scheme, clustertype=clustertype))
        else:
            newcats=set(item.split(';'))
            if newcats.isdisjoint(curcats) and firstitem != 1:      # end of cluster, append cluster length
                csize.append(runlen)
                runlen = 1
            else:                                                   # shared cluster or start of list
                runlen += 1
            
            if clustertype=="fluid":
                curcats = newcats
            elif clustertype=="static":
                curcats = (curcats & newcats)
                if curcats==set([]):
                    curcats = newcats
            else:
                raise ValueError('Invalid cluster type')
        firstitem=0
    csize.append(runlen)
    if sum(csize) > 0:
        clustList += csize
    return clustList

# returns labels in place of items for list or nested lists
# provide list (l) and coding scheme (external file)
def labelClusters(l, scheme, labelIntrusions=False, targetLetter=None):

    """
    Convert items into cluster labels based on a clustering scheme.

    Supports both semantic (dictionary file-based) and letter-based 
    clustering. Optionally labels unknown items as "intrusion".

    Parameters
    ----------
    l : list
        List or nested list of words/items to label.
    scheme : str or int
        Path to a semantic category file (str) or an integer representing the number of letters to use.
    labelIntrusions : bool, optional
        Whether to assign an "intrusion" label to unknown items. Default is False.
    targetLetter : str, optional
        Restricts labeling to items starting with this letter. Only relevant for letter-based clustering.

    Returns
    -------
    list
        A list (or nested list) of labels corresponding to each item.
    """
    ...

    if isinstance(scheme,str):
        clustertype = "semantic"    # reads clusters from a fixed file
    elif isinstance(scheme,int):
        clustertype = "letter"      # if an int is given, use the first N letters as a clustering scheme
        maxletters = scheme
        if targetLetter:
            targetLetter = targetLetter.lower()
        
    else:
        raise Exception('Unknown clustering type in labelClusters()')

    if clustertype == "semantic":
        cf=open(scheme,'rt', encoding='utf-8-sig')
        cats={}
        for line in cf:
            line=line.rstrip()
            if line[0] == "#": continue         # skip commented lines
            cat, item = line.split(',')
            cat=cat.lower().replace(' ','').replace("'","").replace("-","") # basic clean-up
            item=item.lower().replace(' ','').replace("'","").replace("-","")
            if item not in list(cats.keys()):
                cats[item]=cat
            else:
                if cat not in cats[item]:
                    cats[item]=cats[item] + ';' + cat
    labels=[]
    for inum, item in enumerate(l):
        if isinstance(item, list):
            labels.append(labelClusters(item, scheme, labelIntrusions=labelIntrusions, targetLetter=targetLetter))
        else:
            item=item.lower().replace(' ','')
            if clustertype == "semantic":
                if item in list(cats.keys()):
                    labels.append(cats[item])
                elif labelIntrusions:               # if item not in dict, either ignore it or label is as category "intrusion"
                    labels.append("intrusion")
            elif clustertype == "letter":
                if (item[0] == targetLetter) or ((targetLetter == None) and (labelIntrusions == False)):
                    labels.append(item[:maxletters])
                elif labelIntrusions:
                    if targetLetter == None:
                        raise Exception('Cant label intrusions without a target letter [labelClusters]')
                    else:
                         labels.append("intrusion")     # if item not in dict, either ignore it or label is as category "intrusion"
    return labels
