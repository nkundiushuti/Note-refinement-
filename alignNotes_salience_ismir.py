import essentia

import essentia.standard
from essentia.standard import *

import numpy as np
import audiolazy.lazy_midi
from bisect import bisect_left, bisect_right
from scipy import optimize
from scipy import ndimage
from collections import defaultdict
import pickle
import os
import sys
from Graph import Graph
import itertools

from os import listdir
from os.path import isfile, join

# from pylab import plot, show, figure, imshow
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle,Ellipse
# import logging
# # Log everything, and send it to stderr.
# logging.basicConfig(level=logging.DEBUG)

def is_intstring(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def list_duplicates(seq):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items()
                            if len(locs)>0)

def smooth(x,window_len=11,window='hanning'):
        if x.ndim != 1:
                raise ValueError, "smooth only accepts 1 dimension arrays."
        if x.size < window_len:
                raise ValueError, "Input vector needs to be bigger than window size."
        if window_len<3:
                return x
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
        if window == 'flat': #moving average
                w=np.ones(window_len,'d')
        else:
                w=eval('np.'+window+'(window_len)')
        y=np.convolve(w/w.sum(),s,mode='same')
        return y[window_len:-window_len+1]


def gaussian1d(height, center_x, width_x):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    return lambda x: height*np.exp(
                -(((center_x-x)/width_x)**2)/2)

def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def getMidi(instrument,FilePath,suffix,beginTime,finishTime,samplerate,hop,maxAllowedInterval,fixOffset=False,maximumOffset=0.2,melOtherBeginGroup=[],id_inst=0):
    maximumOffset = int(maximumOffset * round(float(samplerate / hop)))
    midifile = FilePath+instrument+suffix+'.txt'
    melodyFromFile = np.genfromtxt(midifile, comments='!', \
      delimiter=',',names="a,b,c",dtype=["f","f","S3"])
    melTimeStampsBegin = melodyFromFile['a'].tolist()
    melTimeStampsEnd = melodyFromFile['b'].tolist()
    startTime1 = bisect_right(melTimeStampsBegin,beginTime)
    #startTime2 = bisect_right(melTimeStampsEnd,beginTime)
    #startTime = np.minimum(startTime1,startTime2)
    startTime = startTime1
    #endTime1 = bisect_left(melTimeStampsBegin,finishTime)
    endTime2 = bisect_left(melTimeStampsEnd,finishTime)
    #endTime = np.maximum(endTime1,endTime2)
    endTime = endTime2
    #print melTimeStampsBegin[startTime]
    #print melTimeStampsEnd[endTime]
    if (startTime<endTime):
        melTimeStampsBegin = melTimeStampsBegin[startTime:endTime]
        melTimeStampsBegin = [x - beginTime for x in melTimeStampsBegin]
        melTimeStampsEnd = melTimeStampsEnd[startTime:endTime]
        melTimeStampsEnd = [x - beginTime for x in melTimeStampsEnd]
        if (melTimeStampsBegin[0] < 0):
           melTimeStampsBegin[0] = 0
        if (melTimeStampsEnd[0] < 0):
           melTimeStampsEnd[0] = 0
        #melTimeStampsBegin[melTimeStampsBegin<0.0] = 0
        #melTimeStampsEnd[melTimeStampsEnd<0.0] = 0
        #get the midi
        melNotesMIDI = melodyFromFile['c'].tolist()
        melNotesMIDI = melNotesMIDI[startTime:endTime]
        if (melTimeStampsBegin[0]==0) and (melTimeStampsEnd[0]==0) :
          melTimeStampsBegin.pop(0)
          melTimeStampsEnd.pop(0)
          melNotesMIDI.pop(0)
        #import pdb; pdb.set_trace()
        melNotesHz = [audiolazy.lazy_midi.midi2freq(audiolazy.lazy_midi.str2midi1(n)) for n in melNotesMIDI]
        melNotesOctave = [audiolazy.lazy_midi.str2octave1(n) for n in melNotesMIDI]
        #freqMin = np.min(melNotesHz)
        #freqMax = np.max(melNotesHz)
        maxAllowedInterval = int(maxAllowedInterval * round(float(samplerate / hop)))
        endMelody = int(melTimeStampsEnd[-1] * float(samplerate / hop))
        melodyBegin = [np.maximum(0,int(mel * float(samplerate / hop))-maxAllowedInterval) for mel in melTimeStampsBegin]
        melodyEnd = [np.minimum(endMelody,(int(mel * float(samplerate / hop))+maxAllowedInterval)) for mel in melTimeStampsEnd]

        #group consecutive notes together
        #get index where each note is played
        melNotesHzGroup=[]
        melodyBeginGroup=[]
        melodyEndGroup=[]
        melNotesOctaveGroup=[]
        melodyIndex=[]

        k=-1
        for dup in sorted(list_duplicates(melNotesMIDI)):
            k=k+1
            melNotesHzGroup.append(audiolazy.lazy_midi.midi2freq(audiolazy.lazy_midi.str2midi1(dup[0])))
            melNotesOctaveGroup.append(audiolazy.lazy_midi.str2octave1(dup[0]))
            melodyBeginGroup.append(np.maximum(0,int(melTimeStampsBegin[dup[1][0]] * float(samplerate / hop))-maxAllowedInterval) )
            melodyEndGroup.append(np.minimum(endMelody,int(melTimeStampsEnd[dup[1][0]] * float(samplerate / hop))+maxAllowedInterval ))
            mIndex = []
            mIndex.append(dup[1][0])
            duration1 = melodyEnd[dup[1][0]]-melodyBegin[dup[1][0]]
            for i in range(1,len(dup[1])):
                duration2 = melodyEnd[dup[1][i]]-melodyBegin[dup[1][i]]
                if (int(melTimeStampsEnd[dup[1][i-1]] * float(samplerate / hop))+maxAllowedInterval)>=np.maximum(0,int(melTimeStampsBegin[dup[1][i]] * float(samplerate / hop))-maxAllowedInterval) \
                and (duration1<(2*duration2)) and (duration2<(2*duration1)):
                    #add to current group
                    melodyEndGroup[k] = np.minimum(endMelody,int(melTimeStampsEnd[dup[1][i]] * float(samplerate / hop))+maxAllowedInterval )
                    mIndex.append(dup[1][i])
                else: #add to new group
                    melodyIndex.append(mIndex)
                    mIndex = []
                    mIndex.append(dup[1][i])
                    k=k+1
                    melNotesHzGroup.append(audiolazy.lazy_midi.midi2freq(audiolazy.lazy_midi.str2midi1(dup[0])))
                    melNotesOctaveGroup.append(audiolazy.lazy_midi.str2octave1(dup[0]))
                    melodyBeginGroup.append(np.maximum(0,int(melTimeStampsBegin[dup[1][i]] * float(samplerate / hop))-maxAllowedInterval) )
                    melodyEndGroup.append(np.minimum(endMelody,int(melTimeStampsEnd[dup[1][i]] * float(samplerate / hop))+maxAllowedInterval ))
            melodyIndex.append(mIndex)

        if fixOffset:
            for i in range(len(melodyBeginGroup)):
                maximumContinuation = 0
                for jj in range(len(melOtherBeginGroup)):
                    for ii in range(len(melOtherBeginGroup[jj])):
                        if (jj!=id_inst) or (i!=ii):
                            if (melodyBeginGroup[i] < melOtherBeginGroup[jj][ii]) and (melodyEndGroup[i] > melOtherBeginGroup[jj][ii]):
                                maximumContinuation = -1
                                ii = len(melOtherBeginGroup[jj])
                                jj = len(melOtherBeginGroup)
                                break
                            elif (maximumContinuation > -1) and (melodyEndGroup[i]<melOtherBeginGroup[jj][ii]):
                                if (maximumContinuation < melOtherBeginGroup[jj][ii]) or (maximumContinuation==0):
                                    maximumContinuation = np.minimum(melodyEndGroup[i]+maximumOffset,melOtherBeginGroup[jj][ii])
                if maximumContinuation == 0:
                    #maximumContinuation = np.minimum(melodyEndGroup[i] + maximumOffset)
                    maximumContinuation = melodyEndGroup[i] + maximumOffset
                if maximumContinuation > 0: #the note can continue further beyond the score offset
                    melodyEndGroup[i] = maximumContinuation
                    melodyEnd[melodyIndex[i][-1]] = maximumContinuation

        #sort the note groups list according to the starting time
        if len(melodyBeginGroup)>1:
            sortedIndex = np.argsort(np.array(melodyBeginGroup))
            melodyBeginGroup = [melodyBeginGroup[i] for i in sortedIndex]
            melodyEndGroup = [melodyEndGroup[i] for i in sortedIndex]
            melNotesHzGroup = [melNotesHzGroup[i] for i in sortedIndex]
            melNotesOctaveGroup = [melNotesOctaveGroup[i] for i in sortedIndex]
            melodyIndex = [melodyIndex[i] for i in sortedIndex]

        return melNotesHz,melodyBegin,melodyEnd,melNotesOctave, \
            melNotesHzGroup,melodyBeginGroup,melodyEndGroup,melNotesOctaveGroup,melodyIndex

    else:
        return [],[],[],[],[],[],[],[],[]

def writeMidi(instrument,FilePath,suffix,notesBegin,notesEnd,beginTime,finishTime,samplerate,hop,id_test,id_test2):
    midifile = FilePath+instrument+suffix+'.txt'
    melodyFromFile = np.genfromtxt(midifile, comments='!', \
      delimiter=',',names="a,b,c",dtype=["f","f","S3"])
    melTimeStampsBegin = melodyFromFile['a'].tolist()
    melTimeStampsEnd = melodyFromFile['b'].tolist()
    startTime = bisect_right(melTimeStampsBegin,beginTime)
    endTime = bisect_left(melTimeStampsEnd,finishTime)

    #get the midi
    melNotesMIDI = melodyFromFile['c'].tolist()
    #convert from frames to time
    nBegin = [beginTime + float(mel * hop) / float(samplerate) for mel in notesBegin]
    nEnd = [beginTime + float(mel * hop) / float(samplerate) for mel in notesEnd]

    #replace the times with the aligned ones, in the chosen time segment
    melTimeStampsBegin[startTime:endTime] = nBegin
    melTimeStampsEnd[startTime:endTime] = nEnd

    #write the csv file
    outfile = FilePath+'aligned/'+instrument+suffix+'_'+str(id_test)+'_'+str(id_test2)+'.txt'
    arr = np.zeros(len(melNotesMIDI), dtype=[('var1','a5'),('var2','a5'),('var3','a5')])
    arr = (np.char.mod('%10.9f', melTimeStampsBegin),np.char.mod('%10.9f', melTimeStampsEnd),[n for n in melNotesMIDI])
    mat = np.transpose(arr)
    np.savetxt(outfile, mat, fmt='%s', delimiter = ',')

def writeF0(instrument,FilePath,suffix,melodyLine,allowedMelodyLines,nsamples,samplerate,hop):
    f0 = np.zeros((nsamples,allowedMelodyLines+1), dtype=float)
    f0[:,0] = [float(i * hop) / float(samplerate) for i in range(nsamples)]
    for i in range(allowedMelodyLines):
        f0[:,i+1] = melodyLine[:,i]

    #write the csv file
    outfile = FilePath+instrument+suffix+'_f0.txt'
    np.savetxt(outfile, f0, fmt='%10.5f', delimiter = ',')

def readMixMatrix(MixingFile,FilePath):
    import csv
    mixMatrix = []
    inst = []
    with open(FilePath+MixingFile, 'r') as data:
        reader = csv.reader(data)
        k = 0
        for row in reader:
            if k==0:
                channelRow = [str(x) for x in row[1:]]
            else:
                numberRow = [float(x) for x in row[1:]] # This slice skips 'date's
                mixMatrix.append(numberRow)
                inst.append(row[0])
            k = k + 1
    mixMatrix = np.array(mixMatrix).astype(np.float)
    return mixMatrix,inst

def readWeights(mixMatrix,instrument,id_inst):
    id_channel = np.argmax(mixMatrix[id_inst,:])
    channel = channelRow[id_channel]
    weights = mixMatrix[:,id_channel]
    return weights,channel

def getHarmonicPartials(f0,maxOctaves=4,maxFreq=5000):
    partials=[]
    c = 1
    f = f0
    while c<(2*maxOctaves) and (f<maxFreq):
        f = c * f0
        partials.append(f)
        c = c + 1
    return partials


###################################################################
#IMAGE PROCESSING
###################################################################
def binarize(img,smooth=False):
    return (img > img.mean()).astype(np.float)

def binarize_local(img,img_b,interval):
    binary = np.zeros_like(img_b)
    down = 0
    up = img.shape[1]/12 + interval - 1
    while (np.sum(img_b[:,up])>6) and (up<img.shape[1]):
            up = up + 1
    while up<img.shape[1]:
        binary[:,down:up] = (img[:,down:up] > img[:,0:up].mean()).astype(np.float)
        down = up + 1
        up = np.minimum(img.shape[1], down + img.shape[1]/12 + interval - 1)
        if (up<img.shape[1]):
            while (up<img.shape[1]) and (np.sum(img_b[:,up])>6):
                if (up<img.shape[1]):
                    up = up + 1
                else:
                    break
    l, nb_l = ndimage.label(binary)
    #detect the blobs that span widely across frequency and split them
    for i in range(1, nb_l+1):
        blobs = ndimage.find_objects(l==i)
        slice_x, slice_y = blobs[0]
        if (slice_y.stop-slice_y.start)>(3*interval):
            steps = int(slice_y.stop-slice_y.start)/int(3*interval)
            for s in range(1,steps):
                binary[:,slice_y.start+s*4*interval:slice_y.start+s*4*interval+4] = 0
            #sums = np.sum(binary[:,slice_y.start:slice_y.stop])
    return binary

def get_blobs_limits(labels, n_labels):
    note_start = np.zeros(n_labels+1)
    note_end = np.zeros(n_labels+1)
    f_start = np.zeros(n_labels+1)
    f_end = np.zeros(n_labels+1)
    for i in range(1, n_labels+1):
        blobs = ndimage.find_objects(labels==i)
        #get the segment where the note blob is
        if len(blobs)>0:
            slice_x, slice_y = blobs[0]
            #get the position of the note
            note_start[i] = slice_x.start
            note_end[i] = slice_x.stop
            f_start[i] = slice_y.start
            f_end[i] = slice_y.stop

    return note_start, note_end, f_start, f_end

def get_blobs_overlapping(note_start_b, note_end_b,note_start_c, note_end_c):
    overlapping_before=np.zeros_like(note_start_b)
    overlapping_current=np.zeros_like(note_start_c)

    # for each pair of blobs
    for i in range(len(note_start_b)):
        for j in range(len(note_start_c)):
            intersect_l = np.maximum(note_start_b[i], note_start_c[j])
            intersect_r = np.minimum(note_end_b[i], note_end_c[j])
            intersect = np.maximum(0, intersect_r-intersect_l)
            if intersect>0: #if there is intersection in time between the blobs
                #increase the overlapping factor for the two blobs
                overlapping_before[i] = np.maximum(overlapping_before[i],intersect/(note_end_b[i]-note_start_b[i]))
                overlapping_current[j] = np.maximum(overlapping_current[j],intersect/(note_end_c[j]-note_start_c[j]))

    return overlapping_before,overlapping_current

def get_blobs_overlapping2d(note_start_b, note_end_b,note_start_c, note_end_c, f_start_b, f_end_b, f_start_c, f_end_c):
    overlapping_before=np.zeros_like(note_start_b)
    overlapping_current=np.zeros_like(note_start_c)

    # for each pair of blobs
    for i in range(len(note_start_b)):
        for j in range(len(note_start_c)):
            intersect_tl = np.maximum(note_start_b[i], note_start_c[j])
            intersect_tr = np.minimum(note_end_b[i], note_end_c[j])
            intersect_t = np.maximum(0, intersect_tr-intersect_tl)
            intersect_fl = np.maximum(f_start_b[i], f_start_c[j])
            intersect_fr = np.minimum(f_end_b[i], f_end_c[j])
            intersect_f = np.maximum(0, intersect_fr-intersect_fl)
            if (intersect_t>0) and (intersect_f>0): #if there is intersection in time between the blobs
                #increase the overlapping factor for the two blobs
                overlapping_before[i] = np.maximum(overlapping_before[i],intersect_t*intersect_f/((note_end_b[i]-note_start_b[i])*(f_end_b[i]-f_start_b[i])))
                overlapping_current[j] = np.maximum(overlapping_current[j],intersect_t*intersect_f/((note_end_c[j]-note_start_c[j])*(f_end_c[j]-f_start_c[j])))

    return overlapping_before,overlapping_current

def get_blobs_combination(note_start, note_end, f_start, f_end, sizes, energy, thresh):
    note_start_new = []
    note_end_new = []
    f_start_new = []
    f_end_new = []
    sizes_new = []
    energy_new = []

    #add the existing blobs
    for k in range(len(note_start)):
        note_start_new.append(note_start[k])
        note_end_new.append(note_end[k])
        f_start_new.append(f_start[k])
        f_end_new.append(f_end[k])
        sizes_new.append(sizes[k])
        energy_new.append(energy[k])

    #add the combinations of blobs
    combos = itertools.combinations(range(1,len(note_start)), 2)
    # for each pair of blobs
    for c in combos:
        i = c[0]
        j = c[1]
        combinen = False
        intersect_l = np.maximum(note_start[i], note_start[j])
        intersect_r = np.minimum(note_end[i], note_end[j])
        intersect = np.maximum(0, intersect_r-intersect_l)
        #if there is no intersection in time between the blobs
        if (intersect==0):
            if ((0.7*(note_end[j]-note_start[j]))>abs(intersect_r-intersect_l)) and ((0.7*(note_end[i]-note_start[i]))>abs(intersect_r-intersect_l)):
                #combine the two blobs to create a new one
                if note_end[i] < note_start[j]:
                    ns=note_start[i]
                    ne=note_end[j]
                else:
                    ns=note_start[j]
                    ne=note_end[i]
                if (note_end[i]-note_start[i])>(note_end[j]-note_start[j]):
                    fs=f_start[i]
                    fe=f_end[i]
                else:
                    fs=f_start[j]
                    fe=f_end[j]
                s=sizes[i]+sizes[j]
                e=energy[i]+energy[j]
                combinen = True
        #if blobs overlap
        else:
            if abs(intersect_r-intersect_l)<(0.7*(note_end[i]-note_start[i])):
                if note_end[i]>intersect_l:
                    ns=note_start[i]
                    ne=intersect_l
                else:
                    ns=intersect_r
                    ne=note_end[i]
                fs=f_start[i]
                fe=f_end[i]
                s=sizes[i]-(abs(intersect_r-intersect_l)/(note_end[i]-note_start[i]))*sizes[i]
                e=energy[i]-(abs(intersect_r-intersect_l)/(note_end[i]-note_start[i]))*energy[i]
                combinen = True
            else :
                if abs(intersect_r-intersect_l)<(0.7*(note_end[j]-note_start[j])):
                    if note_end[j]>intersect_l:
                        ns=note_start[j]
                        ne=intersect_l
                    else:
                        ns=intersect_r
                        ne=note_end[j]
                    fs=f_start[j]
                    fe=f_end[j]
                    s=sizes[j]-(abs(intersect_r-intersect_l)/(note_end[j]-note_start[j]))*sizes[j]
                    e=energy[j]-(abs(intersect_r-intersect_l)/(note_end[j]-note_start[j]))*energy[j]
                    combinen = True

            if (combinen==True):
                #if (len(abs((note_end-note_start)-(ne-ns))>))>0)
                similar1 = np.extract(abs(np.array(note_start)-ns)<thresh, note_start)
                similar2 = np.extract(abs(np.array(note_end)-ne)<thresh, note_end)
                if (len(similar1)<1) or (len(similar2)<1):
                    note_start_new.append(ns)
                    note_end_new.append(ne)
                    f_start_new.append(fs)
                    f_end_new.append(fe)
                    sizes_new.append(s)
                    energy_new.append(e)


    return np.array(note_start_new), np.array(note_end_new), np.array(f_start_new), np.array(f_end_new), np.array(sizes_new), np.array(energy_new)



################################################################
# MAIN FUNCTION
################################################################

def computePeaks(FilePath,Mixfile,mixingMatrix,instrument_list,suffix, timeSpan):
    ###############
    #initialization
    ###############
    interval = 70 #filter the spectral peaks in double this interval(100=semitone)
    # use just this part of midi score files
    #beginTime = 450
    #finishTime = 465
    # beginTime = 420
    # finishTime = 480
    beginTime = 0
    finishTime = 50
    #resolution for hte salience function
    binResolution = 10
    f0 = 110
    hopSize = 256
    frameSize = 4096
    # hopSize = 128
    # frameSize = 2048
    sampleRate = 44100
    maximumOffset = 0.8 #(seconds) allow a note to continue if there is no other note playing
    ploti = 0
    framek = 55

    #general audio processing algorithms: windowing,spectrum
    w = Windowing(type='blackmanharris92', zeroPadding=3*frameSize)
    spectrum = Spectrum(size=frameSize)  # FFT() would return the complex FFT, here we just want the magnitude spectrum
    hpeaks = HarmonicPeaks() #find the harmonic peaks given the spectral peaks and the fundamental frequency
    speaks = SpectralPeaks(magnitudeThreshold=0,maxPeaks=300,minFrequency=105)
    #whitening = SpectralWhitening()

    #get mixing matrix and all the instruments in the mixture
    if mixingMatrix==True:
        MixingFile = 'Panning_matrix.csv'
        mixMatrix,all_instruments = readMixMatrix(MixingFile,FilePath)
        instruments = all_instruments
    else:
        instruments = instrument_list
        weights = [1.0/float(len(instrument_list)) for isx in instrument_list]
        Filename = Mixfile
        audioFile = FilePath+Filename+'.wav'
        loader = essentia.standard.EqloudLoader(filename = audioFile)
        audio = loader()
        #compute the spectrum for each frame
        spec = []
        fqn = []
        mgn = []
        for frame in FrameGenerator(audio, frameSize = frameSize, hopSize = hopSize):
            s = spectrum(w(frame)) #compute the spectrum
            spec.append(s)
            #compute the spectral peaks
            f,m = speaks(s)
            #m = whitening(s,f,m)
            maskf = np.ones(f.shape, dtype=np.bool)
            maskf[f<=0] = False #remove negative or zero peaks
            fqn.append(f[maskf])
            mgn.append(m[maskf])

    mNotesHzI=[]
    mBeginI=[]
    mEndI=[]
    mNotesHzI=[]
    mNotesOctaveI=[]
    mBeginIg=[]
    mEndIg=[]
    mNotesHzIg=[]
    allF = []
    allM = []
    for instrument in instrument_list:
        id_inst = instruments.index(instrument)
        if mixingMatrix==True:
            weights,channel = readWeights(mixMatrix,id_inst)
            Filename = channel
            audioFile = FilePath+Filename+'.wav'
            loader = essentia.standard.EqloudLoader(filename = audioFile)
            audio = loader()
            #compute the spectrum for each frame
            spec = []
            fqn = []
            mgn = []
            for frame in FrameGenerator(audio, frameSize = frameSize, hopSize = hopSize):
                s = spectrum(w(frame)) #compute the spectrum
                spec.append(s)
                #compute the spectral peaks
                f,m = speaks(s)
                #m = whitening(s,f,m)
                maskf = np.ones(f.shape, dtype=np.bool)
                maskf[f<=0] = False #remove negative or zero peaks
                fqn.append(f[maskf])
                mgn.append(m[maskf])


        #get the midi for the instrument to be aligned
        melodyNotesHzI,melodyBeginI,melodyEndI,melodyNotesOctaveI,melNotesHzGroup,melodyBeginGroup,melodyEndGroup,melNotesOctaveGroup,melodyIndex = getMidi(instrument,FilePath,suffix,beginTime,finishTime,sampleRate,hopSize,timeSpan,False)
        mNotesHzI.append(melodyNotesHzI);mBeginI.append(melodyBeginI);mEndI.append(melodyEndI);mNotesOctaveI.append(melodyNotesOctaveI)
        #get the groundtruth for the instrument to be aligned,for plotting purposes
        melodyNotesHzIg,melodyBeginIg,melodyEndIg,melodyNotesOctaveIg,melNotesHzGroupg,melodyBeginGroupg,melodyEndGroupg,melNotesOctaveGroupg,melodyIndexg = getMidi(instrument,FilePath,'_g',beginTime,finishTime,sampleRate,hopSize,0,False)
        mNotesHzIg.append(melodyNotesHzIg);mBeginIg.append(melodyBeginIg);mEndIg.append(melodyEndIg)

        #check to see if there are notes in the score that pass the size of audio
        for i in range(len(melodyEndGroup)):
            if melodyEndGroup[i] > int(np.floor(len(audio)/hopSize)):
                melodyEndI[melodyIndex[i][-1]] = int(np.floor(len(audio)/hopSize))
                melodyEndGroup[i] = int(np.floor(len(audio)/hopSize))

        noteF =[]
        noteM = []
        #for every note compute spectral peaks, spectral salience
        for i in range(len(melodyNotesHzI)):
            #octa gives the harmonic partials of the current note
            octa = getHarmonicPartials(melodyNotesHzI[i])
            fmin = octa[0] - (octa[0] * (2.0**(interval/1200.0)) - octa[0]) #minimum frequency

            #create a vector to save frequencies and magnitudes for every frame
            filtered_fqn = [[] for idx in range(melodyEndI[i]-melodyBeginI[i])]
            filtered_mgn = [[] for idx in range(melodyEndI[i]-melodyBeginI[i])]

            for k in range(melodyEndI[i]-melodyBeginI[i]):

                f = fqn[k + melodyBeginI[i]]
                m = mgn[k + melodyBeginI[i]]
                # mask = [np.where((f<=(o+o * (2.0**(interval/1200.0)))) & (f>=(o-o * (2.0**(interval/1200.0))))) for o in octa]
                # filtered_fqn[k].append(filf[kdx])
                # filtered_mgn[k].append(film[kdx])

                #filter the peaks for this note and for all its harmonic partials
                #foundHarmonics = np.zeros(len(octa),dtype=int)
                for oidx,o in enumerate(octa):  #for all the harmonic partials
                    fband = o * (2.0**(interval/1200.0)) - o  #get the frequency band for this partial
                    mask = np.ones(len(f), dtype=np.bool)
                    mask[f<=(o-fband)] = False #filter the peaks below the interval
                    mask[f>=(o+fband)] = False #filter the peaks up the interval
                    filf=f[mask]
                    film=m[mask]
                    #find the harmonic peaks given the spectral peaks and the fundamental frequency
                    #vhfilf,hfilm = hpeaks(filf,film,melodyNotesHzI[i])
                    if len(filf) > 0:
                        #foundHarmonics[oidx] = 1 #we've found peaks for these harmonic partials
                        for kdx in range(len(filf)):
                            #??penalize the deviation from the center frequency by multiplying with a factor??
                            filtered_fqn[k].append(filf[kdx])
                            filtered_mgn[k].append(film[kdx])
                # #if we found peaks for most of the harmonics
                # if np.sum(foundHarmonics)>=(len(foundHarmonics)/2):
                #     #fill in the missing harmonics with the corresponding peaks and magnitude values
                #     for oidx in numpy.where(foundHarmonics == 0)[0]:
                #         meanF = 0
                #         for oidy in numpy.where(foundHarmonics == 1)[0]:
                #             if (octa[oidy]>octa[oidx]):
                #                 meanF = meanF + octa[oidy] / float(np.floor(octa[oidy]/octa[oidx]))
                #             else:
                #                 meanF = meanF + octa[oidy] * float(np.floor(octa[oidx]/octa[oidy]))
                #         meanF =  meanF / np.sum(foundHarmonics)
                #         filtered_fqn[k].append(meanF)
                #         filtered_mgn[k].append(film[kdx])

                # if (i==ploti) and (k==framek) and (id_inst==0):
                #     maxamp = np.amax(np.array(mgn[framek]))
                #     currentAxis = plt.gca()
                #     plt.plot(f,m,'og')
                #     plt.plot(filtered_fqn[framek],filtered_mgn[framek],'-or')
            #save the frequencies and magnitudes for this note
            noteF.append(filtered_fqn)
            noteM.append(filtered_mgn)
        #save the frequencies and magnitudes for this instrument
        allF.append(noteF)
        allM.append(noteM)

    if not os.path.exists(FilePath+'saved/'):
        os.makedirs(FilePath+'saved/')
    with open(FilePath+'saved/spectral_peaks_'+Filename+suffix+'.pickle', 'w') as f:
        pickle.dump([allF,allM], f)

    print "saved spectral peaks for file:"+Filename+suffix




def computeSalience(FilePath,Mixfile,mixingMatrix,instrument_list,suffix,id_instrument_to_align,id_test, timeSpan):
    ###############
    #initialization
    ###############
    interval = 70 #filter the spectral peaks in double this interval(100=semitone)
    # use just this part of midi score files
    #beginTime = 450
    #finishTime = 465
    # beginTime = 420
    # finishTime = 480
    beginTime = 0
    finishTime = 50
    #resolution for the salience function
    binResolution = 10
    f0 = 110
    hopSize = 256
    frameSize = 4096
    # hopSize = 128
    # frameSize = 2048
    sampleRate = 44100
    maximumOffset = 0.8 #(seconds) allow a note to continue if there is no other note playing
    ploti = 0
    framek = 55

    #get mixing matrix and all the instruments in the mixture
    if mixingMatrix==True:
        MixingFile = 'Panning_matrix.csv'
        mixMatrix,all_instruments = readMixMatrix(MixingFile,FilePath)
        instruments = all_instruments
    else:
        instruments = instrument_list
        weights = [1.0/float(len(instrument_list)) for isx in instrument_list]
        Filename = Mixfile
        audioFile = FilePath+Filename+'.wav'
        loader = essentia.standard.EqloudLoader(filename = audioFile)
        audio = loader()

    mNotesHzI=[]
    mBeginI=[]
    mEndI=[]
    mNotesHzI=[]
    mNotesOctaveI=[]
    mBeginIg=[]
    mEndIg=[]
    mNotesHzIg=[]
    for instrument in instrument_list:
        id_inst = instruments.index(instrument)
        if mixingMatrix==True:
            weights,channel = readWeights(mixMatrix,id_inst)
            Filename = channel
            audioFile = FilePath+Filename+'.wav'
            loader = essentia.standard.EqloudLoader(filename = audioFile)
            audio = loader()

        #get the midi for the instrument to be aligned
        melodyNotesHzI,melodyBeginI,melodyEndI,melodyNotesOctaveI,melNotesHzGroup,melodyBeginGroup,melodyEndGroup,melNotesOctaveGroup,melodyIndex = getMidi(instrument,FilePath,suffix,beginTime,finishTime,sampleRate,hopSize,timeSpan,False)
        mNotesHzI.append(melodyNotesHzI);mBeginI.append(melodyBeginI);mEndI.append(melodyEndI);mNotesOctaveI.append(melodyNotesOctaveI)
        #get the groundtruth for the instrument to be aligned,for plotting purposes
        melodyNotesHzIg,melodyBeginIg,melodyEndIg,melodyNotesOctaveIg,melNotesHzGroupg,melodyBeginGroupg,melodyEndGroupg,melNotesOctaveGroupg,melodyIndexg = getMidi(instrument,FilePath,'_g',beginTime,finishTime,sampleRate,hopSize,0,False)
        mNotesHzIg.append(melodyNotesHzIg);mBeginIg.append(melodyBeginIg);mEndIg.append(melodyEndIg)

        #check to see if there are notes in the score that pass the size of audio
        for i in range(len(melodyEndGroup)):
            if melodyEndGroup[i] > int(np.floor(len(audio)/hopSize)):
                melodyEndI[melodyIndex[i][-1]] = int(np.floor(len(audio)/hopSize))
                melodyEndGroup[i] = int(np.floor(len(audio)/hopSize))

        #get the maximum f0 played in the score of all the instruments
        maxf=110
        for n in range(len(mNotesHzI)):
            mNHzI = mNotesHzI[n]
            for k in range(len(mNHzI)):
                if (maxf<mNHzI[k]):
                    maxf=mNHzI[k]


    ###############
    # computation
    ###############

    if os.path.isfile(FilePath+'saved/spectral_peaks_'+Filename+suffix+'.pickle'):
        with open(FilePath+'saved/spectral_peaks_'+Filename+suffix+'.pickle') as f:
            allF,allM = pickle.load(f)

    #now we computed the peaks for an instrument and each note
    #we take each 2 notes and we decrease the magnitude of the common peaks in the overlapping zones
    #for instidx,instrument in enumerate(instrument_list):
    instidx = id_instrument_to_align
    instrument = instrument_list[id_instrument_to_align]

    melodyNotesHzI=mNotesHzI[instidx];melodyBeginI=mBeginI[instidx];melodyEndI=mEndI[instidx]
    melodyNotesHzIg=mNotesHzIg[instidx];melodyBeginIg=mBeginIg[instidx];melodyEndIg=mEndIg[instidx]
    blobs_salience=[]
    for i in range(len(melodyNotesHzI)):
        filtered_fqn = allF[instidx][i]
        filtered_mgn = allM[instidx][i]
        # f = filtered_fqn[framek]
        # m = filtered_mgn[framek]
        #octa gives the harmonic partials of the current note
        octa = getHarmonicPartials(melodyNotesHzI[i])
        fbands = [o * (2.0**(interval/1200.0)) - o for o in octa]
        fmin = octa[0] - (octa[0] * (2.0**(interval/1200.0)) - octa[0]) #minimum frequency
        #decrease the magnitude for the peaks of the other overlapping notes from all instruments
        overlap_factor = [[0 for f_idx in range(len(filtered_fqn[overlap_index]))] for overlap_index in range(melodyEndI[i]-melodyBeginI[i])]
        for n,other_inst in enumerate(instruments):
            if other_inst in instrument_list:
                other_id = instrument_list.index(other_inst)
                mNHzI = mNotesHzI[other_id]; mBI = mBeginI[other_id]; mEI = mEndI[other_id]
            else:
                mNHzI = melodyNotesHz[n]; mBI = melodyBegin[n]; mEI = melodyEnd[n]
            for k in range(len(mNHzI)): #all the notes from EVERY instrument
                if (n != id_inst) or (i != k) or (melodyNotesHzI[i]!=mNHzI[k]): #if it's note the current note from the current instrument, or the same midi note
                    #if there is overlapping in time between the current note and the overlapping note
                    intersect_l = np.maximum(melodyBeginI[i], mBI[k])
                    intersect_r = np.minimum(melodyEndI[i], mEI[k])
                    if np.maximum(0, intersect_r-intersect_l)>0: #if there is intersection in time
                        #for all the overlapping frames in the overlapping note
                        for j in range(intersect_l-melodyBeginI[i],intersect_r-melodyBeginI[i]):
                            oct = getHarmonicPartials(mNHzI[k])
                            if other_inst in instrument_list:
                                other_id = instrument_list.index(other_inst)
                                j_other = j + melodyBeginI[i] - mBI[k]
                                other_fqn = allF[other_id][k]
                                other_mgn = allM[other_id][k]
                                wanted = set(other_fqn[j_other])
                                indices =[idx for (idx, value) in enumerate(filtered_fqn[j]) if value in wanted]
                                for ind in indices:
                                    closestF = min(enumerate(octa), key=lambda x: abs(x[1]-filtered_fqn[j][ind]))
                                    fband = fbands[closestF[0]]
                                    harmonic = np.round(filtered_fqn[j][ind] / octa[0]) # the number of harmonic

                                    if (id_test==0):
                                        gamp=1.0 #the amplitude of the gaussian
                                        factor = weights[n]
                                    elif (id_test==4):
                                        gamp = 0.8**(harmonic-1)
                                        factor = weights[n]
                                    elif (id_test==1):
                                        gamp = 0.8**(harmonic-1)
                                        factor = (1-weights[n])
                                    elif (id_test==2):
                                        gamp=1.0 #the amplitude of the gaussian
                                        factor = (melodyNotesHzI[i]/maxf)*(1-weights[n])
                                    elif (id_test==3):
                                        gamp = 0.8**(harmonic-1)
                                        factor = (melodyNotesHzI[i]/maxf)*(1-weights[n])
                                    elif (id_test==5):
                                        harmonic_other = np.round(filtered_fqn[j][ind]  / oct[0]) #the number of harmonic with respect to the overlapping note
                                        #sum up this overlapping
                                        overlap_factor[j][ind] = overlap_factor[j][ind] + 0.8**(harmonic_other-1)*weights[id_inst]


                                    if (id_test!=5):
                                        g = gaussian1d(gamp,closestF[1],fband/2)
                                        filtered_mgn[j][ind] = filtered_mgn[j][ind] * factor * g(filtered_fqn[j][ind])

        if (id_test==5):
            for j in range(melodyEndI[i]-melodyBeginI[i]):
                for ind in range(len(filtered_mgn[j])):
                    if overlap_factor[j][ind]>0:
                        f_closer = min(octa, key=lambda x:abs(x-filtered_fqn[j][ind]))
                        fband = f_closer * (2.0**(interval/1200.0)) - f_closer
                        g = gaussian1d(1.0,f_closer,fband/2)
                        filtered_mgn[j][ind] = filtered_mgn[j][ind] * np.minimum(0.01,1.0-overlap_factor[j][ind]) * g(filtered_fqn[j][ind])


        # if (i==ploti):
        #     for oidx,o in enumerate(octa):
        #         fband = o * (2.0**(interval/1200.0)) - o
        #         currentAxis.add_patch(Rectangle((o-fband, 0), 2*fband, maxamp, facecolor="red", alpha=0.3, edgecolor="none",linewidth=0.08))
        #     plt.plot(filtered_fqn[framek],filtered_mgn[framek],'-ob')
        #     plt.show()
        #create a salience vector for the current note
        sal = np.zeros((melodyEndI[i]-melodyBeginI[i], 5*int(1200 / binResolution)))
        # sal1 = np.zeros((melodyEndI[i]-melodyBeginI[i]+1, 5*int(1200 / binResolution)))

        #compute the pitch salience for this note
        psalience = PitchSalienceFunction(binResolution=binResolution,referenceFrequency=fmin,harmonicWeight=0.8,magnitudeThreshold=40,numberHarmonics=10)
        for k in range(melodyEndI[i]-melodyBeginI[i]):
            s = psalience(essentia.array(filtered_fqn[k]),essentia.array(filtered_mgn[k]))
            sal[k,0:] = essentia.array(s).T
            # s1 = psalience(essentia.array(allF[instidx][i][k]),essentia.array(allM[instidx][i][k]))
            # sal1[k,0:] = essentia.array(s1).T
            # import pdb;pdb.set_trace()

        blobs_salience.append(sal)

    with open(FilePath+'saved/blobs_'+instrument+suffix+'_'+str(id_test)+'.pickle', 'w') as f:
        pickle.dump(blobs_salience, f)
    print "saved salience: "+instrument+". test case: "+str(id_test)


def alignNotes(FilePath,Mixfile,mixingMatrix,instrument_list,suffix,id_instrument_to_align,id_test,id_test2, timeSpan):
    ###############
    #initialization
    ###############
    interval = 70 #filter the spectral peaks in double this interval(100=semitone)
    # use just this part of midi score files
    #beginTime = 450
    #finishTime = 465
    # beginTime = 420
    # finishTime = 480
    beginTime = 0
    finishTime = 50
    #resolution for hte salience function
    binResolution = 10
    f0 = 110
    hopSize = 256
    frameSize = 4096
    # hopSize = 128
    # frameSize = 2048
    sampleRate = 44100
    maximumOffset = 0.8 #(seconds) allow a note to continue if there is no other note playing
    ploti = 0
    framek = 55
    ov_weight = 0.7
    aov_weight = [0.4,0.6,0.8,1]
    binsInOctave = 1200.0 / binResolution;
    # if id_test2<2:
    #     dijk = False
    # else:
    #     dijk = True
    dijk = True

    #get mixing matrix and all the instruments in the mixture
    if mixingMatrix==True:
        MixingFile = 'Panning_matrix.csv'
        mixMatrix,all_instruments = readMixMatrix(MixingFile,FilePath)
        instruments = all_instruments
    else:
        instruments = instrument_list
        weights = [1.0/float(len(instrument_list)) for isx in instrument_list]
        Filename = Mixfile
        audioFile = FilePath+Filename+'.wav'
        loader = essentia.standard.EqloudLoader(filename = audioFile)
        audio = loader()

    mNotesHzI=[]
    mBeginI=[]
    mEndI=[]
    mNotesHzI=[]
    mNotesOctaveI=[]
    mBeginIg=[]
    mEndIg=[]
    mNotesHzIg=[]
    for instrument in instrument_list:
        id_inst = instruments.index(instrument)
        if mixingMatrix==True:
            weights,channel = readWeights(mixMatrix,id_inst)
            Filename = channel
            audioFile = FilePath+Filename+'.wav'
            loader = essentia.standard.EqloudLoader(filename = audioFile)
            audio = loader()

        #get the midi for the instrument to be aligned
        melodyNotesHzI,melodyBeginI,melodyEndI,melodyNotesOctaveI,melNotesHzGroup,melodyBeginGroup,melodyEndGroup,melNotesOctaveGroup,melodyIndex = getMidi(instrument,FilePath,suffix,beginTime,finishTime,sampleRate,hopSize,timeSpan,False)
        mNotesHzI.append(melodyNotesHzI);mBeginI.append(melodyBeginI);mEndI.append(melodyEndI);mNotesOctaveI.append(melodyNotesOctaveI)
        #get the groundtruth for the instrument to be aligned,for plotting purposes
        melodyNotesHzIg,melodyBeginIg,melodyEndIg,melodyNotesOctaveIg,melNotesHzGroupg,melodyBeginGroupg,melodyEndGroupg,melNotesOctaveGroupg,melodyIndexg = getMidi(instrument,FilePath,'_g',beginTime,finishTime,sampleRate,hopSize,0,False)
        mNotesHzIg.append(melodyNotesHzIg);mBeginIg.append(melodyBeginIg);mEndIg.append(melodyEndIg)

        #check to see if there are notes in the score that pass the size of audio
        for i in range(len(melodyEndGroup)):
            if melodyEndGroup[i] > int(np.floor(len(audio)/hopSize)):
                melodyEndI[melodyIndex[i][-1]] = int(np.floor(len(audio)/hopSize))
                melodyEndGroup[i] = int(np.floor(len(audio)/hopSize))


    ###############
    # computation
    ###############

    instidx = id_instrument_to_align
    instrument = instrument_list[id_instrument_to_align]

    melodyNotesHzI=mNotesHzI[instidx];melodyBeginI=mBeginI[instidx];melodyEndI=mEndI[instidx]
    melodyNotesHzIg=mNotesHzIg[instidx];melodyBeginIg=mBeginIg[instidx];melodyEndIg=mEndIg[instidx]

    if os.path.isfile(FilePath+'saved/blobs_'+instrument+suffix+'_'+str(id_test)+'.pickle'):
        with open(FilePath+'saved/blobs_'+instrument+suffix+'_'+str(id_test)+'.pickle') as f:
            blobs_salience = pickle.load(f)
    else:
        print 'pickle file could not be found'


    len_best_path = 0


    #absolute time
    noteStartI = [0 for s in melodyBeginI]
    noteEndI = [0 for s in melodyEndI]
    #relative time
    note_start = [0 for s in melodyBeginI]
    note_end = [0 for s in melodyEndI]
    f_start = [0 for s in melodyBeginI]
    f_end = [0 for s in melodyEndI]
    binarized = []
    labels = []
    nlabels = []
    best_blob = []
    bestmax_blobs = []
    total_energy=[]
    total_size=[]
    NS_all = []
    NE_all = []
    FS_all = []
    FE_all = []
    ov_before = [[] for r in range(len(melodyNotesHzI))]
    ov_after = [[] for r in range(len(melodyNotesHzI))]
    ov_before2d = [[] for r in range(len(melodyNotesHzI))]
    ov_after2d = [[] for r in range(len(melodyNotesHzI))]

    nVertices = 0
    noteVertices = []
    blobVertices = []
    idxVertices = []
    nBlobs = []
    edges = []
    idxBlobs = []

    #compute the score for every blob
    for i in range(len(melodyNotesHzI)):

        sal = np.array(blobs_salience[i])
        #import pdb;pdb.set_trace()
        #binarize the salience image
        sal_b = binarize(sal,False)
        salb = binarize_local(sal,sal_b,interval/binResolution)
        # salb = ndimage.binary_erosion(salb)
        # # Remove small white regions
        # salb = ndimage.binary_opening(salb)
        # # Remove small black hole
        # salb = ndimage.binary_closing(salb)

        # #label all blobs
        # l, nb_l = ndimage.label(salb)
        # sizes_c = ndimage.sum(salb, l, range(0,nb_l+1))
        # #remove blobs with area smaller than 100ms X 'interval' semitone bins=100X7=700 pixels
        # mask_size = sizes_c <  ((interval/binResolution)**2)
        # remove_pixel = mask_size[l]
        # salb[remove_pixel] = 0
        l, nb_l = ndimage.label(salb)

        #get the boundaries of all the blobs
        note_start_all,note_end_all,f_start_all,f_end_all = get_blobs_limits(l, nb_l)
        energy_c = ndimage.sum(sal, l, range(0,nb_l+1))
        sizes_c = ndimage.sum(salb, l, range(0,nb_l+1))


        # if id_test2==2:
        #     note_start_combo, note_end_combo, f_start_combo, f_end_combo, sizes_combo, energy_combo = get_blobs_combination(note_start_all, note_end_all, f_start_all, f_end_all, sizes_c, energy_c, int(0.02*np.floor(float(sampleRate/hopSize))))
        #     total_energy.append(energy_combo)
        #     total_size.append(sizes_combo)
        #     NS_all.append(note_start_combo)
        #     NE_all.append(note_end_combo)
        #     FS_all.append(f_start_combo)
        #     FE_all.append(f_end_combo)
        # else:
        total_energy.append(energy_c)
        total_size.append(sizes_c)
        NS_all.append(note_start_all)
        NE_all.append(note_end_all)
        FS_all.append(f_start_all)
        FE_all.append(f_end_all)

        labels.append(l)
        nlabels.append(nb_l)
        binarized.append(salb)

        # #measure the overlapping between blobs of this note and blobs of the next note
        # if id_test2<4:
        #     if i==0:
        #         ov_before[i] = np.zeros_like(note_start_all)
        #     elif i>0:
        #         overlapping_before,overlapping_current = get_blobs_overlapping(melodyBeginI[i-1]+NS_all[i-1],melodyBeginI[i-1]+NE_all[i-1],melodyBeginI[i]+NS_all[i],melodyBeginI[i]+NE_all[i])
        #         ov_before[i] = overlapping_current
        #         ov_after[i-1] = overlapping_before
        #     if i==(len(melodyNotesHzI)-1):
        #         ov_after[i] = np.zeros_like(note_start_all)
        # elif id_test2>3:
        #     if i==0:
        #         ov_before2d[i] = np.zeros_like(note_start_all)
        #     elif i>0:
        #         #make the 2 salience matrices relative to the same octave
        #         fmin_b = melodyNotesHzI[i-1] - (melodyNotesHzI[i-1] * (2.0**(interval/1200.0)) - melodyNotesHzI[i-1]) #minimum frequency
        #         fmin_c = melodyNotesHzI[i] - (melodyNotesHzI[i] * (2.0**(interval/1200.0)) - melodyNotesHzI[i]) #minimum frequency
        #         fref_b = (fmin_b//np.minimum(fmin_b,fmin_c))*np.minimum(fmin_b,fmin_c)
        #         fref_c = (fmin_c//np.minimum(fmin_b,fmin_c))*np.minimum(fmin_b,fmin_c)
        #         referenceTerm_b = 0.5 - binsInOctave * np.log2(fref_b)
        #         referenceTerm_c = 0.5 - binsInOctave * np.log2(fref_c)
        #         f_bins_b = np.floor(binsInOctave * np.log2(fmin_b) + referenceTerm_b)
        #         f_bins_c = np.floor(binsInOctave * np.log2(fmin_c) + referenceTerm_c)
        #         overlapping_before2d,overlapping_current2d = get_blobs_overlapping2d(melodyBeginI[i-1]+NS_all[i-1], melodyBeginI[i-1]+NE_all[i-1],melodyBeginI[i]+NS_all[i], melodyBeginI[i]+NE_all[i], f_bins_b+FS_all[i-1], f_bins_b+FE_all[i-1], f_bins_c+FS_all[i], f_bins_c+FE_all[i])
        #         ov_before2d[i] = overlapping_current2d
        #         ov_after2d[i-1] = overlapping_before2d
        #     if i==(len(melodyNotesHzI)-1):
        #         ov_after2d[i] = np.zeros_like(note_start_all)


    #compute the best blobs according to the score
    for i in range(len(melodyNotesHzI)):
        sal = np.array(blobs_salience[i])
        salb = binarized[i]
        energy_c = total_energy[i]
        sizes_c = total_size[i]
        l = labels[i]
        nb_l = nlabels[i]
        note_start_all = NS_all[i]
        note_end_all = NE_all[i]
        f_start_all = FS_all[i]
        f_end_all = FE_all[i]

        # #compute the weighted score for each blob, considering the overlapping
        # ov_weight = aov_weight[id_test2%4]
        # if id_test2<4:
        #     weighted1 = np.array([sizes_c[tk]*(1-ov_weight*(ov_after[i][tk]-ov_before[i][tk])) for tk in range(len(sizes_c))])
        # elif id_test2>3:
        #     weighted2 = np.array([sizes_c[tk]*(1-ov_weight*(ov_after2d[i][tk]-ov_before2d[i][tk])) for tk in range(len(sizes_c))])


        #keep only onsets in the interval [0,0.6s], where we know that the actual onset is
        filtert1 = np.where(note_start_all<=int(2.0*(timeSpan+0.1)*np.floor(float(sampleRate/hopSize))))[0]
        # #keep only notes which are larger than 70ms
        minduration=np.minimum(int(0.07*np.floor(float(sampleRate/hopSize))),np.amax(note_end_all-note_start_all))
        filtert2 = np.where((note_end_all-note_start_all)>=minduration)[0]
        filtert = np.intersect1d(filtert1,filtert2)
        # #eliminate blobs which span a huge frequency range
        filterf = np.where((f_end_all-f_start_all)<50)[0]
        # filters = np.intersect1d(filtert,filterf)
        filters = np.intersect1d(filtert,filterf)
        #filters = filterf
        #remove zero because index 0 is the whole image
        filters = filters[filters != 0]



        #build Dijkstra graph nodes with the filtered blobs
        selectedb = filters
        if (len(selectedb)>0):
            if (i==0) or (i==(len(melodyNotesHzI)-1)):
                idxVertices.append(nVertices)
                nVertices = nVertices + 1
                nBlobs.append(1)
                selectedb = np.array([filters[np.argmax(energy_c[filters])]])
                idxBlobs.append(selectedb)
                noteVertices.extend([i])
                blobVertices.extend(selectedb)
            else:
                idxVertices.append(nVertices)
                nVertices = nVertices + len(selectedb)
                nBlobs.append(len(selectedb))
                idxBlobs.append(selectedb)
                noteVertices.extend([i for tind in range(len(selectedb))])
                blobVertices.extend(selectedb)
        else: #there is no blob in the filtered list, so pick the best one
            idxVertices.append(nVertices)
            nVertices = nVertices + 1
            nBlobs.append(1)
            selectedb = np.array([np.argmax(energy_c[1:])+1])
            idxBlobs.append(selectedb)
            noteVertices.extend([i])
            blobVertices.extend(selectedb)

        #construct the edges, compute the cost between each blobs of this note and each blob of the note before
        if i>0:
            noEdge = True
            # w1 = np.ones_like(total_energy[i-1]) - total_energy[i-1] / np.amax(total_energy[i-1])
            # w2 = np.ones_like(total_energy[i]) - total_energy[i] / np.amax(total_energy[i])
            w1 = np.ones_like(total_size[i-1]) - total_size[i-1] / np.amax(total_size[i-1])
            w2 = np.ones_like(total_size[i]) - total_size[i] / np.amax(total_size[i])
            for b1 in range(nBlobs[i-1]):
                for b2 in range(nBlobs[i]):
                    s1=melodyBeginI[i-1] + NS_all[i-1][idxBlobs[i-1][b1]]
                    e1=melodyBeginI[i-1] + NE_all[i-1][idxBlobs[i-1][b1]]
                    s2=melodyBeginI[i] + NS_all[i][idxBlobs[i][b2]]
                    e2=melodyBeginI[i] + NE_all[i][idxBlobs[i][b2]]
                    if e2>s1:
                        intersect_l = np.maximum(s1, s2)
                        intersect_r = np.minimum(e1, e2)
                        p1 = 0
                        p2 = 0
                        penalization = 1
                        if np.maximum(0, intersect_r-intersect_l)>0: #if there is intersection in time between the two blobs
                            p1 = (intersect_r-intersect_l)/(e1-s1)
                            p2 = (intersect_r-intersect_l)/(e2-s2)
                            penalization=penalization+1.1*(p1+p2)
                            #costb = (1+p1) * w1[idxBlobs[i-1][b1]] + (1+p2) * w2[idxBlobs[i][b2]]
                            #costb = (1+(p1+p2)*0.5)*(w1[idxBlobs[i-1][b1]] + w2[idxBlobs[i][b2]])
                            #costb = p1+p2
                        if (p1<0.7) and (p2<0.7):
                            costb = penalization*w1[idxBlobs[i-1][b1]] + w2[idxBlobs[i][b2]]
                            edges.append([idxVertices[i-1]+b1,idxVertices[i]+b2,costb])
                            noEdge=False
            #have maximum a solution, so build at least 1 edge
            if noEdge==True:
                print 'no edge',i
                for b1 in range(nBlobs[i-1]):
                    for b2 in range(nBlobs[i]):
                        s1=melodyBeginI[i-1] + NS_all[i-1][idxBlobs[i-1][b1]]
                        e1=melodyBeginI[i-1] + NE_all[i-1][idxBlobs[i-1][b1]]
                        s2=melodyBeginI[i] + NS_all[i][idxBlobs[i][b2]]
                        e2=melodyBeginI[i] + NE_all[i][idxBlobs[i][b2]]
                        if e2>s1:
                            intersect_l = np.maximum(s1, s2)
                            intersect_r = np.minimum(e1, e2)
                            p1 = 0
                            p2 = 0
                            penalization = 1
                            if np.maximum(0, intersect_r-intersect_l)>0: #if there is intersection in time between the two blobs
                                p1 = (intersect_r-intersect_l)/(e1-s1)
                                p2 = (intersect_r-intersect_l)/(e2-s2)
                                penalization=penalization+1.1*(p1+p2)

                            costb = penalization*w1[idxBlobs[i-1][b1]] + w2[idxBlobs[i][b2]]
                            edges.append([idxVertices[i-1]+b1,idxVertices[i]+b2,costb])
                            noEdge=False
            if noEdge==True:
                print 'still no edge',i

        # #get the best blob, the simple way e.g. without dynamic programming
        # if id_test2<4:
        #     if len(filters)>0:
        #         best = filters[np.argmax(weighted1[filters])]
        #     else:
        #         best = np.argmax(weighted1[1:]) + 1
        # elif id_test2>3:
        #     if len(filters)>0:
        #         best = filters[np.argmax(weighted2[filters])]
        #     else:
        #         best = np.argmax(weighted2[1:]) + 1
        # # else:
        # if id_test2==0:
        if len(filters)>0:
            bestmax = filters[np.argmax(energy_c[filters])]
        else:
            bestmax = np.argmax(energy_c[1:]) + 1
        bestmax_blobs.append(bestmax)


    #compute the best path in the blob's graph
    V=range(nVertices)
    m=Graph(V,edges)
    print "size of graph is ", m.size
    best_path = m.dijkstra(0,nVertices-1)
    len_best_path = len(best_path[1])
    print 'best path ', len_best_path
    # temp_b = [noteVertices[idxx] for idxx in best_path[1]]
    # print temp_b
    #print "distance and best route is", best_path
    if len_best_path>0:
        for idx in best_path[1]:
            i = noteVertices[idx]
            bestb = blobVertices[idx]
            bestmax = bestmax_blobs[i]

            if (np.abs(NS_all[i][bestb]-NS_all[i][bestmax])>int(0.07*np.floor(float(sampleRate/hopSize)))) or \
            (np.abs(NE_all[i][bestb]-NE_all[i][bestmax])>int(0.07*np.floor(float(sampleRate/hopSize)))):

                best_blob.append(bestmax)

                blobs = ndimage.find_objects(l==bestmax)
                note_start[i] = NS_all[i][bestmax]
                note_end[i] = NE_all[i][bestmax]
                f_start[i] = FS_all[i][bestmax]
                f_end[i] = FE_all[i][bestmax]

                noteStartI[i] = melodyBeginI[i] + note_start[i]
                noteEndI[i] = melodyBeginI[i] + note_end[i]

                # plt.subplot(121)
                # imshow(np.array(blobs_salience[i]).T, origin='lower')
                # plt.axvline(x=(melodyBeginIg[i]-melodyBeginI[i]), linewidth=1, color='r')
                # plt.axvline(x=(melodyEndIg[i]-melodyBeginI[i]), linewidth=1, color='r')
                # plt.axvline(x=(noteStartI[i]-melodyBeginI[i]), linewidth=1, color='b')
                # plt.axvline(x=(noteEndI[i]-melodyBeginI[i]), linewidth=1, color='b')
                # plt.subplot(122)
                # imshow(np.array(binarized[i]).T, cmap=plt.cm.spectral, origin='lower')
                # plt.axvline(x=(melodyBeginIg[i]-melodyBeginI[i]), linewidth=1, color='r')
                # plt.axvline(x=(melodyEndIg[i]-melodyBeginI[i]), linewidth=1, color='r')
                # plt.axvline(x=(noteStartI[i]-melodyBeginI[i]), linewidth=1, color='b')
                # plt.axvline(x=(noteEndI[i]-melodyBeginI[i]), linewidth=1, color='b')
                # c = labels[i]==bestb
                # plt.contour(c.astype(int).T, [0.5], linewidths=2, colors='green')
                # c = labels[i]==bestmax
                # plt.contour(c.astype(int).T, [0.5], linewidths=2, colors='blue')
                # plt.show()

            else:
                best_blob.append(bestb)

                note_start[i] = NS_all[i][bestb]
                note_end[i] = NE_all[i][bestb]
                f_start[i] = FS_all[i][bestb]
                f_end[i] = FE_all[i][bestb]

                noteStartI[i] = melodyBeginI[i] + note_start[i]
                noteEndI[i] = melodyBeginI[i] + note_end[i]

                # plt.subplot(121)
                # imshow(np.array(blobs_salience[i]).T, origin='lower')
                # plt.axvline(x=(melodyBeginIg[i]-melodyBeginI[i]), linewidth=1, color='r')
                # plt.axvline(x=(melodyEndIg[i]-melodyBeginI[i]), linewidth=1, color='r')
                # plt.axvline(x=(noteStartI[i]-melodyBeginI[i]), linewidth=1, color='b')
                # plt.axvline(x=(noteEndI[i]-melodyBeginI[i]), linewidth=1, color='b')
                # plt.subplot(122)
                # imshow(np.array(binarized[i]).T, cmap=plt.cm.spectral, origin='lower')
                # plt.axvline(x=(melodyBeginIg[i]-melodyBeginI[i]), linewidth=1, color='r')
                # plt.axvline(x=(melodyEndIg[i]-melodyBeginI[i]), linewidth=1, color='r')
                # plt.axvline(x=(noteStartI[i]-melodyBeginI[i]), linewidth=1, color='b')
                # plt.axvline(x=(noteEndI[i]-melodyBeginI[i]), linewidth=1, color='b')
                # c = labels[i]==bestb
                # plt.contour(c.astype(int).T, [0.5], linewidths=2, colors='green')
                # plt.show()
    else:
        len_best_path = len(melodyNotesHzI)
        for i in range(len(melodyNotesHzI)):
            bestmax = bestmax_blobs[i]
            best_blob.append(bestmax)

            blobs = ndimage.find_objects(l==bestmax)
            #get the segment where the note blob is
            if len(blobs)>0:
                slice_x, slice_y = blobs[0]
                #get the position of the note
                note_start[i] = slice_x.start
                note_end[i] = slice_x.stop
                f_start[i] = slice_y.start
                f_end[i] = slice_y.stop
            else:
                note_start[i] = 0
                note_end[i] = melodyEndI[i]-melodyBeginI[i]
                f_start[i] = 0
                f_end[i] = 2*interval/binResolution


            noteStartI[i] = melodyBeginI[i] + slice_x.start
            noteEndI[i] = melodyBeginI[i] + slice_x.stop

            # #histogram of the starting/ending times
            # hist_start_c, bin_edges_start_c = np.histogram(note_start_all, density=True)
            # hist_end_c, bin_edges_end_c = np.histogram(note_end_all, density=True)

            # plt.subplot(121)
            # left,right = bin_edges_start_c[:-1],bin_edges_start_c[1:]
            # X_start_c = np.array([left,right]).T.flatten()
            # Y_start_c = np.array([hist_start_c,hist_start_c]).T.flatten()
            # left,right = bin_edges_end_c[:-1],bin_edges_end_c[1:]
            # X_end_c = np.array([left,right]).T.flatten()
            # Y_end_c = np.array([hist_end_c,hist_end_c]).T.flatten()
            # plt.plot(X_start_c,Y_start_c, color='b')
            # plt.plot(X_end_c,Y_end_c, color='g')
            # plt.axvline(x=(melodyBeginIg[i]-melodyBeginI[i]), linewidth=1, color='r')
            # plt.axvline(x=(melodyEndIg[i]-melodyBeginI[i]), linewidth=1, color='r')
            # plt.subplot(122)
            # imshow(salb.T, cmap=plt.cm.spectral, origin='lower')
            # plt.axvline(x=(melodyBeginIg[i]-melodyBeginI[i]), linewidth=1, color='r')
            # plt.axvline(x=(melodyEndIg[i]-melodyBeginI[i]), linewidth=1, color='r')
            # plt.show()




    #see if there are any notes intersecting
    for i in range(len(melodyNotesHzI)):
        sal_c = np.array(blobs_salience[i])
        note_start_c = note_start[i]
        note_end_c = note_end[i]

        #if the current note overlaps with the next one, then find the minimum point to correct both of them
        if i<(len(melodyNotesHzI)-1):

            #get the next note
            sal_a = np.array(blobs_salience[i+1])
            note_start_a = note_start[i+1]
            note_end_a = note_end[i+1]

            d = (melodyBeginI[i] + note_end_c)-(melodyBeginI[i+1] + note_start_a)
            if d>0:
                if d>4:
                    patch_c = blobs_salience[i][note_start_c:,np.maximum(0,f_start[i]):f_end[i]]
                    patch_a = blobs_salience[i+1][0:note_end_a,np.maximum(0,f_start[i+1]):f_end[i+1]]
                    apatch_c = blobs_salience[i][:,np.maximum(0,f_start[i]):f_end[i]]
                    apatch_a = blobs_salience[i+1][:,np.maximum(0,f_start[i+1]):f_end[i+1]]
                    t = 0.1
                    dt = d
                    # err_c = 0
                    # err_a = 0
                    note_start_tc = note_start_c
                    note_end_tc = note_end_c
                    note_start_ta = note_start_a
                    note_end_ta = note_end_a
                    tt=0
                    while (dt>0) and (t<2):
                        patch_cb = (patch_c > (t*apatch_c.mean())).astype(np.float)
                        patch_ab = (patch_a > (t*apatch_a.mean())).astype(np.float)
                        l_c,nl_c = ndimage.label(patch_cb)
                        l_a,nl_a = ndimage.label(patch_ab)
                        # anote_start_tc,anote_end_tc,af_start_tc,af_end_tc = get_blobs_limits(l_c, nl_c)
                        # anote_start_ta,anote_end_ta,af_start_ta,af_end_ta = get_blobs_limits(l_a, nl_a)
                        # if (len(anote_end_tc)>1) and (len(anote_start_ta)>1):
                        #     best_c = np.argmax(anote_end_tc[1:])
                        #     best_a = np.argmin(anote_start_ta[1:])

                        # if t in [0.3,0.5,0.7,1,1.2,1.4]:
                        #     plt.subplot(320+tt)
                        #     imshow(np.array(patch_cb).T, cmap=plt.cm.spectral, origin='lower')
                        #     plt.subplot(320+tt+1)
                        #     imshow(np.array(patch_ab).T, cmap=plt.cm.spectral, origin='lower')
                        #     tt = tt + 2
                        if (patch_c.shape[0]>0) and (patch_c.shape[1]>0):
                            e_c = ndimage.sum(patch_c, l_c, range(0,nl_c+1))
                            if nl_c>0:
                                w_c = e_c[1:]
                                best_c = np.argmax(w_c)+1
                                b_c = ndimage.find_objects(l_c==best_c)
                                if len(b_c)>0:
                                    slice_xc, slice_yc = b_c[0]
                                    #get the position of the note
                                    if slice_xc.stop>(0.1*np.floor(float(sampleRate/hopSize))):
                                        note_start_tc = note_start_c+slice_xc.start
                                        note_end_tc = note_start_c+slice_xc.stop

                        if (patch_a.shape[0]>0) and (patch_a.shape[1]>0):
                            e_a = ndimage.sum(patch_a, l_a, range(0,nl_a+1))
                            if nl_a>0:
                                w_a = e_a[1:]
                                best_a = np.argmax(w_a)+1
                                b_a = ndimage.find_objects(l_a==best_a)
                                if len(b_a)>0:
                                    slice_xa, slice_ya = b_a[0]
                                    #get the position of the note
                                    if (slice_xa.stop-slice_xa.start)>(0.1*np.floor(float(sampleRate/hopSize))):
                                        note_start_ta = slice_xa.start
                                        note_end_ta = slice_xa.stop

                        dt = (melodyBeginI[i] + note_end_tc) - (melodyBeginI[i+1] + note_start_ta)
                        if dt<=0:
                            # if (id_test2%2==1):
                            ratioa=(note_start_ta-note_start_a)/(melodyBeginI[i] + note_end_c) - (melodyBeginI[i+1] + note_start_a)
                            ratioc=(note_end_c-note_end_tc)/(melodyBeginI[i] + note_end_c) - (melodyBeginI[i+1] + note_start_a)
                            note_end_c=note_end_tc+ratioc/(ratioc+ratioa)*(-1)*dt
                            note_start_a=note_start_ta-ratioa/(ratioc+ratioa)*(-1)*dt
                            note_start[i+1] = note_start_a
                            note_end[i] = note_end_c
                            noteStartI[i+1] = melodyBeginI[i+1] + note_start_a
                            noteEndI[i] = melodyBeginI[i] + note_end_c
                            t=2
                            #     # if dt<-4:
                            #     #     if abs(abs(note_end_c-note_start_tc)-err_c)<abs(abs(note_end_a-note_start_ta)-err_a):
                            #     #         note_end_c= melodyBeginI[i+1] - melodyBeginI[i] + note_start_ta
                            #     #         note_start_a=note_start_ta
                            #     #     else:
                            #     #         note_end_c=note_end_tc
                            #     #         note_start_a=melodyBeginI[i]+note_end_tc-melodyBeginI[i+1]
                            # else:
                            #     note_end_c=note_end_tc
                            #     note_start_a=note_start_ta
                            #     note_start[i+1] = note_start_a
                            #     note_end[i] = note_end_c
                            #     noteStartI[i+1] = melodyBeginI[i+1] + note_start_a
                            #     noteEndI[i] = melodyBeginI[i] + note_end_c
                            #     t=2
                        else:
                            t=t+0.1
                            if (t==2) and (dt<d):
                                note_end_c=note_end_tc
                                note_start_a=note_start_ta
                                note_start[i+1] = note_start_a
                                note_end[i] = note_end_c
                                noteStartI[i+1] = melodyBeginI[i+1] + note_start_a
                                noteEndI[i] = melodyBeginI[i] + note_end_c
                                dt = (melodyBeginI[i] + note_end_tc) - (melodyBeginI[i+1] + note_start_ta)
                        # if (id_test2%2==1):
                        #     # if (dt<0):
                        #     #     split = ((melodyBeginI[i] + note_end_c) + (melodyBeginI[i+1] + note_start_a)) / 2
                        #     #     # if split>melodyEndI[i]:
                        #     #     #     split = melodyEndI[i]
                        #     #     # elif split<melodyBeginI[i+1]:
                        #     #     #     split = melodyBeginI[i+1]
                        #     #     note_end_c=split-melodyBeginI[i]
                        #     #     note_start_a=melodyBeginI[i+1]-split
                        #     #     note_start[i+1] = note_start_a
                        #     #     note_end[i] = note_end_c
                        #     #     noteStartI[i+1] = melodyBeginI[i+1] + note_start_a
                        #     #     noteEndI[i] = melodyBeginI[i] + note_end_c
                        #     #save the error
                        #     err_c = abs(note_end_c-note_start_tc)
                        #     err_a = abs(note_start_ta-note_start_a)
                        #     # else:
                        #     #     t=t+0.1
                    #plt.show()

                else:
                    noteEndI[i] = noteStartI[i+1]

            # #if (abs(noteStartI[i]-melodyBeginIg[i])>20) or (abs(noteStartI[i+1]-melodyBeginIg[i+1])>20):
            # if (abs(noteStartI[i+1]-melodyBeginIg[i+1])>20) or (abs(noteEndI[i]-melodyEndIg[i])>20):
            # #if (abs(noteStartI[i+1]-melodyBeginIg[i+1])>20):
            #     print 'i',i+1
            #     if melodyNotesHzI[i]==melodyNotesHzI[i+1]:
            #         print 'same note'
            #     plt.subplot(161)
            #     imshow(np.array(blobs_salience[i]).T, origin='lower')
            #     plt.axvline(x=(melodyBeginIg[i]-melodyBeginI[i]), linewidth=1, color='r')
            #     plt.axvline(x=(melodyEndIg[i]-melodyBeginI[i]), linewidth=1, color='r')
            #     plt.axvline(x=(noteStartI[i]-melodyBeginI[i]), linewidth=1, color='y')
            #     plt.axvline(x=(noteEndI[i]-melodyBeginI[i]), linewidth=1, color='y')
            #     plt.axvline(x=(melodyBeginIg[i+1]-melodyBeginI[i]), linewidth=1, color='g')
            #     plt.subplot(162)
            #     imshow(np.array(binarized[i]).T, cmap=plt.cm.spectral, origin='lower')
            #     plt.axvline(x=(melodyBeginIg[i]-melodyBeginI[i]), linewidth=1, color='r')
            #     plt.axvline(x=(melodyEndIg[i]-melodyBeginI[i]), linewidth=1, color='r')
            #     plt.axvline(x=(noteStartI[i]-melodyBeginI[i]), linewidth=1, color='y')
            #     plt.axvline(x=(noteEndI[i]-melodyBeginI[i]), linewidth=1, color='y')
            #     plt.axvline(x=(melodyBeginIg[i+1]-melodyBeginI[i]), linewidth=1, color='g')
            #     c = labels[i]==best_blob[i]
            #     plt.contour(c.astype(int).T, [0.5], linewidths=2, colors='green')
            #     plt.subplot(163)
            #     imshow(np.array(blobs_salience[i+1]).T, origin='lower')
            #     plt.axvline(x=(melodyBeginIg[i+1]-melodyBeginI[i+1]), linewidth=1, color='r')
            #     plt.axvline(x=(melodyEndIg[i+1]-melodyBeginI[i+1]), linewidth=1, color='r')
            #     plt.axvline(x=(noteStartI[i+1]-melodyBeginI[i+1]), linewidth=1, color='y')
            #     plt.axvline(x=(noteEndI[i+1]-melodyBeginI[i+1]), linewidth=1, color='y')
            #     plt.axvline(x=(melodyEndIg[i]-melodyBeginI[i+1]), linewidth=1, color='g')
            #     plt.subplot(164)
            #     imshow(np.array(binarized[i+1]).T, cmap=plt.cm.spectral, origin='lower')
            #     plt.axvline(x=(melodyBeginIg[i+1]-melodyBeginI[i+1]), linewidth=1, color='r')
            #     plt.axvline(x=(melodyEndIg[i+1]-melodyBeginI[i+1]), linewidth=1, color='r')
            #     plt.axvline(x=(noteStartI[i+1]-melodyBeginI[i+1]), linewidth=1, color='y')
            #     plt.axvline(x=(noteEndI[i+1]-melodyBeginI[i+1]), linewidth=1, color='y')
            #     plt.axvline(x=(melodyEndIg[i]-melodyBeginI[i+1]), linewidth=1, color='g')
            #     c = labels[i+1]==best_blob[i+1]
            #     plt.contour(c.astype(int).T, [0.5], linewidths=2, colors='green')
            #     if (d>4):
            #         plt.subplot(165)
            #         # patch_c = blobs_salience[i][:,np.maximum(0,f_start[i]):f_end[i]]
            #         # patch_cb = binarize(patch_c)
            #         imshow(patch_cb.T, cmap=plt.cm.spectral, origin='lower')
            #         plt.axvline(x=(melodyBeginIg[i]-melodyBeginI[i]), linewidth=1, color='r')
            #         plt.axvline(x=(melodyEndIg[i]-melodyBeginI[i]), linewidth=1, color='r')
            #         plt.axvline(x=(noteStartI[i]-melodyBeginI[i]), linewidth=1, color='w')
            #         plt.axvline(x=(noteEndI[i]-melodyBeginI[i]), linewidth=1, color='w')
            #         plt.axvline(x=(melodyBeginIg[i+1]-melodyBeginI[i]), linewidth=1, color='g')
            #         plt.subplot(166)
            #         # patch_a = blobs_salience[i+1][:,np.maximum(0,f_start[i+1]):f_end[i+1]]
            #         # patch_ab = binarize(patch_a)
            #         imshow(patch_ab.T, cmap=plt.cm.spectral, origin='lower')
            #         plt.axvline(x=(melodyBeginIg[i+1]-melodyBeginI[i+1]), linewidth=1, color='r')
            #         plt.axvline(x=(melodyEndIg[i+1]-melodyBeginI[i+1]), linewidth=1, color='r')
            #         plt.axvline(x=(noteStartI[i+1]-melodyBeginI[i+1]), linewidth=1, color='w')
            #         plt.axvline(x=(noteEndI[i+1]-melodyBeginI[i+1]), linewidth=1, color='w')
            #         plt.axvline(x=(melodyEndIg[i]-melodyBeginI[i+1]), linewidth=1, color='g')
            #     plt.show()

    if not os.path.exists(FilePath+'aligned/'):
        os.makedirs(FilePath+'aligned/')
    writeMidi(instrument,FilePath,suffix,noteStartI,noteEndI,beginTime,finishTime,sampleRate,hopSize,id_test,id_test2)
    #writeMidi(instrument,FilePath,suffix,noteStartI,noteEndI,beginTime,finishTime,sampleRate,hopSize,1,2)
    print "saved alignment for: "+instrument+". test case: "+str(id_test)+"_"+str(id_test2)


if __name__=='__main__':

    if len(sys.argv)>1:
        if not is_intstring(sys.argv[1]):
            stage = 0
        else:
            stage = int(sys.argv[1])
    else:
        stage=2

    if len(sys.argv)>2:
        if not is_intstring(sys.argv[2]):
            case = 0
        else:
            case = int(sys.argv[2])
            case = case - 1
    else:
        case = 0
    print case
    if len(sys.argv)>3:
        dirPath = sys.argv[3]
    else:
        dirPath = '/Volumes/Macintosh HD 2/Documents/Database/Bach10/Sources/'

    #print dirPath
    dirlist = os.walk(dirPath).next()[1]
    dirlist = sorted(dirlist)

    #id of the file in Bach10 to align 0-10
    id_file = 0
    FilePath = dirPath + str(dirlist[id_file]) + '/'
    Mixfile = str(dirlist[id_file])

    instrument_list = ['bassoon','clarinet','saxophone','violin']
    #id of the instrument to align 0..3
    id_instrument_to_align = 0

    suffixes = ['_d','_b']
    #score alignment output to align :0 or 1 (d or b)
    id_suffix =0

    #time span where to look for the notes
    timeSpan = 0.2 #add this value to the offset and subtract it from onset


    #best settings for alignment method
    id_test = 1
    id_test2 = 2


    computePeaks(FilePath,Mixfile,False,instrument_list,suffixes[id_suffix], timeSpan)
    computeSalience(FilePath,Mixfile,False,instrument_list,suffixes[id_suffix],id_instrument_to_align,id_test, timeSpan)
    alignNotes(FilePath,Mixfile,False,instrument_list,suffixes[id_suffix],id_instrument_to_align,id_test,id_test2, timeSpan)






