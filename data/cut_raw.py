import os
import fnmatch
import soundfile as sf

dist_folder = './cut_raw'

def find_cut_point(txt_path):
    file = open(txt_path, 'r')

    min_interval = 1500000
    phn_timing = []
    try:
        text_lines = file.readlines()
        len_line = len(text_lines)
        for i, line in enumerate(text_lines):
            line = line.replace('\n', '')
            [start, end, phn] = line.split(' ')
            start = int(start)
            end = int(end)
            interval = end - start
            if phn == 'None' and interval > min_interval:
                if start == 0:
                    #  first pau
                    phn_timing.append([start, end-min_interval, 'cut'])
                    phn_timing.append([end - min_interval, end, phn])
                elif i == len_line-1:
                    #  last pau
                    phn_timing.append([start, start + min_interval, phn])
                    phn_timing.append([start + min_interval, end, 'cut'])
                else:
                    phn_timing.append([start, start + min_interval, phn])
                    phn_timing.append([start + min_interval, end - min_interval, 'cut'])
                    phn_timing.append([end - min_interval, end, phn])

            else:
                if phn == 'sil' or phn == 'pau':
                    phn = 'cut'
                phn_timing.append([start, end, phn])

    finally:
        file.close()


    return phn_timing

def cut_txt(phn_timing, fname, dist_folder):

    txts = []
    time_pairs = []
    time_pair = [None, None]

    for i, line in enumerate(phn_timing):
        [start, end, phn] = line
        if phn == 'cut':
            if None not in time_pair:
                pass
            else:
                if time_pair[0] == None:
                    pass
                else:
                    time_pair[1] = start
                    if time_pair[0] != time_pair[1]:
                        time_pairs.append(time_pair)
                    time_pair = [None, None]
                time_pair[0] = end
                txts.append([])
        else:
            txts[-1].append(line)

    clean_txt = []
    for i in txts:
        if len(i)>0:
            clean_txt.append(i)

    #处理多余的pau
    clean_txt_trim=[]
    time_pair_trim=[]
    assert len(clean_txt) == len(time_pairs)
    for i in range(len(clean_txt)):
        if len(clean_txt[i]) == 1 and clean_txt[i][0][-1] == 'pau':
            continue
        clean_txt_trim.append(clean_txt[i])
        time_pair_trim.append(time_pairs[i])


    for i, item in enumerate(clean_txt_trim):
        new_path = os.path.join(dist_folder, file_name+'_'+str(i)+'.lab')
        file = open(new_path, 'w')
        start_num = 0
        for i, line in enumerate(item):
            [start, end, phn] = line
            if i ==0:
                start_num = start
            start -= start_num
            end -= start_num
            line = str(start)+' '+str(end)+' '+phn
            file.write(line)
            file.write("\n")
        file.close()


    return time_pair_trim


def cut_wav(wav_path, time_pair, fname, dist_folder):

    [start, stop] = time_pair
    start = int(int(start)*48000/10000000)
    stop = int(int(stop)*48000/10000000)

    y, osr = sf.read(wav_path, subtype='PCM_16', channels=1, samplerate=48000,
                     endian='LITTLE', start=start, stop=stop)

    new_path = os.path.join(dist_folder,fname+'.raw')
    sf.write(new_path, y, subtype='PCM_16', samplerate=48000, endian='LITTLE')

if __name__ == '__main__':

    if not os.path.exists(dist_folder):
        os.mkdir(dist_folder)

    raw_folder = './raw'

    supportedExtensions = '*.raw'
    for dirpath, dirs, files in os.walk(raw_folder):
        for file in fnmatch.filter(files, supportedExtensions):

            file_name = file.replace('.raw','')
            raw_path = os.path.join(dirpath, file)
            txt_path = raw_path.replace('.raw', '.lab')

            phn_timing = find_cut_point(txt_path)
            time_pairs = cut_txt(phn_timing, file_name, dist_folder)

            for i, tp in enumerate(time_pairs):
                fname = file_name+'_'+str(i)
                cut_wav(raw_path, tp, fname, dist_folder)
