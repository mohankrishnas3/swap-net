import sys
import os
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2

# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/mk/Downloads/swap-net-master/swap-net-summer/RAKE-tutorial-master')
sys.path.insert(1, '/home/mk/Downloads/swap-net-master/swap-net-summer/gensim-develop')
sys.path.insert(1, '/home/mk/Downloads/swap-net-master/swap-net-summer/')


import rake
import gensim
import re



# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

article_start = '<a>'
article_end = '</a>'

w_start = '<w>'
w_end = '</w>'

_DIGIT_RE = re.compile(br"\s\d+\s|\s\d+$")




cnn_summary_dir="dataset/neuralsum/cnn"
dm_summary_dir="dataset/neuralsum/dailymail"

cnn_tokenized_stories_dir = "dataset/cnn_data_tokenized"
dm_tokenized_stories_dir = "dataset/dm_data_tokenized"
finished_files_dir = "dataset/finished_files/"
chunks_dir = os.path.join(finished_files_dir, "chunked")




CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data


_DIGIT_RE = re.compile(br"\s\d+\s|\s\d+$")
SYB_RE=re.compile(b"([.,!?\"':;)(])|--")


_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_NUL = b"_NUL"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK,_NUL]

#_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

def chunk_file(set_name):
  in_file = os.path.join(finished_files_dir ,'%s.bin' % set_name)
  reader = open(in_file, "rb")
  chunk = 0
  finished = False
  while not finished:
    chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
    with open(chunk_fname, 'wb') as writer:
      for _ in range(CHUNK_SIZE):
        len_bytes = reader.read(8)
        if not len_bytes:
          finished = True
          break
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, example_str))
      chunk += 1


def chunk_all():
  # Make a dir to hold the chunks
  if not os.path.isdir(chunks_dir):
    os.mkdir(chunks_dir)
  # Chunk the data
  for set_name in ['train', 'val', 'test']:
  #for set_name in [ 'test']:
    print ("Splitting %s data into chunks...",set_name)
    chunk_file(set_name)
  print("Saved chunked data in %s",chunks_dir)


def tokenize_stories(stories_dir, tokenized_stories_dir):
  """Maps a whole directory of .story files to a tokenized version using Stanford CoreNLP Tokenizer"""
  print ("Preparing to tokenize",stories_dir," to ",tokenized_stories_dir)
  stories = os.listdir(stories_dir)
  # make IO list file
  print("Making list of files to tokenize...")
  with open("mapping.txt", "w") as f:
    for s in stories:
      f.write("%s \t %s\n" % (os.path.join(stories_dir, s), os.path.join(tokenized_stories_dir, s)))
  command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', 'mapping.txt']
  print("Tokenizing", len(stories)," files in", stories_dir, " and saving in", tokenized_stories_dir)
  subprocess.call(command)
  print("Stanford CoreNLP Tokenizer has finished.")
  os.remove("mapping.txt")

  # Check that the tokenized stories directory contains the same number of files as the original directory
  num_orig = len(os.listdir(stories_dir))
  num_tokenized = len(os.listdir(tokenized_stories_dir))
#  if num_orig != num_tokenized:
#    raise Exception("The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
  print("Successfully finished tokenizing ",stories_dir," to tokenized_stories_dir.\n")


def read_text_file(text_file):
  lines = []
  with open(text_file, "r") as f:
    for line in f:
      lines.append(line.strip())
  return lines








def prepare_key_vocab():
  _START_VOCAB=[_PAD, _GO, _EOS, _UNK, _NUL]
  """Fill input queue with ModelInput."""
  #vocab_path = os.path.join(FLAGS.train_path, "vocab%d.txt" % FLAGS.vocabulary_size)
  key_path = os.path.join(finished_files_dir, "keyword" )
  vocab_path = os.path.join(finished_files_dir, "vocab" )
  vocab_size=150000
  if os.path.exists(vocab_path):
    vocab = []
    vocab.extend(_START_VOCAB)
    f=open(vocab_path)
    for line in f:
      v=line.strip().split()

      if len(v)==2:
        vocab.append(v[0])
        if len(vocab)==vocab_size : 
          break
  if os.path.exists(key_path):
    key = []
    f=open(key_path)
    for line in f:
      k=line.strip()
      
      if k in vocab:
        key.append(k)
    #print(key)
    vocabulary_path= os.path.join(finished_files_dir, "vocab%d.txt" % vocab_size)
    keyword_path=os.path.join(finished_files_dir, "keyword%d.txt" % vocab_size)
    with open(vocabulary_path, mode="wb") as vocab_file:
      for w in vocab:
        
                
        w = bytes(str(w), 'utf-8') # modified
        #print(w)
        vocab_file.write((w + b"\n"))

    with open(keyword_path, mode="wb") as key_file:
      for w in key:
        
        w = bytes(str(w), 'utf-8') # modified
        key_file.write(w + b"\n")
    #vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return key
  else:
    print("keyword file %s not found.", key_path)



def prepare_key():
  """Fill input queue with ModelInput."""
  #vocab_path = os.path.join(FLAGS.train_path, "vocab%d.txt" % FLAGS.vocabulary_size)
  key_path = os.path.join(finished_files_dir, "keyword150000.txt" )
#  key_path = os.path.join(finished_files_dir, "keyword" )
  if os.path.exists(key_path):
    key = []
    with open(key_path) as f:
      key.extend(f.readlines())
    key = [line.strip() for line in key]
    #vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return key
  else:
    print("keyword file %s not found.", key_path)





def get_everything(story_file,key):
  #lines = read_text_file(story_file)
  f=open(story_file)
  

  lines = f.read().split('\n\n')
 
  
  sent_label=[]
  article_lines=[]
  word_extract=[]
  w=[]

  for line in lines[1].split('\n'):
    if line =="":
      continue
    line = line.strip().lower().split()
    sent_label.append(line[-1])
    sent_one=' '.join(line[:-1])

    sent_one = bytes(sent_one, 'utf-8') # modified
    sent_one=_DIGIT_RE.sub(b" 0",sent_one)
    #article_lines.append(' '.join(line[:-1]))
    article_lines.append(sent_one)

  for line in lines[2].split('\n'):
    if line =="":
      continue
    line= line.strip().lower().replace('*','')
    

    w+=[i for i in line.split() if (i  in key or bool(re.match('@entity'r'\d',i)))]
     
    word_extract.append(line)

  w_set=set(w)
  word_label=[]
  for line in article_lines:
    wd=[1 if k in w else 0 for k in line.strip().split()]
    word_label.append(' '.join(["%s" % (sent) for sent in wd]))

  for line in lines[3].split('\n'):
    if line =="":
      continue
    line = line.strip().lower().split(':')
    entity_0=line[0].strip()
    entity_1=' '.join(line[1:]).strip()

  abstract_new=lines[2].replace('\n', ' <eos> ').replace('*',' ')
  

 
  article_text = ' '.join(["%s %s %s" % (article_start, sent, article_end) for sent in article_lines])
  
  word_label_text=' '.join(["%s %s %s" % (w_start, sent, w_end)  for sent in word_label])
  word_text=' '.join(["%s" % ( sent) for sent in w])

  sent_label_text=' '.join(["%s" % (sent) for sent in sent_label])



  return article_text,sent_label_text,word_text,word_label_text,abstract_new


def get_art_abs(story_file):
  #lines = read_text_file(story_file)
  f=open(story_file)
 
  lines = f.read().split('\n\n')
  

  sent_label=[]
  article_lines=[]
  word_extract=[]
  abstract=[]

  for line in lines[1].split('\n'):
    if line =="":
      continue
    line = line.strip().lower().split()
    sent_label.append(line[-1])
    article_lines.append(' '.join(line[:-1]))

  for line in lines[2].split('\n'):
    if line =="":
      continue
    line= line.strip().lower().replace('*','')
    word_extract.append(line)

  for line in lines[3].split('\n'):
    if line =="":
      continue
    line = line.strip().lower().split(' : ')
    entity_0=line[0]
    entity_1=' '.join(line[1:])

  abstract_new=lines[2].replace('\n', ' <eos> ').replace('*',' ')
  
  return article_lines, abstract, word_extract

def make_vocab(data_file):
  vocab_counter = collections.Counter()
  extract=""

  data_list = read_text_file(data_file)

  vocab_counter = collections.Counter()

  for idx,s in enumerate(data_list):
    if os.path.isfile(os.path.join(cnn_tokenized_stories_dir, s)):
      story_file = os.path.join(cnn_tokenized_stories_dir, s)
      
    elif os.path.isfile(os.path.join(dm_tokenized_stories_dir, s)):
      story_file = os.path.join(dm_tokenized_stories_dir, s)
      

    article_lines,abstract,word_extract_lines=get_art_abs(story_file)

    

    article=' '.join([line for line in article_lines])
    word_extract=' '.join([line for line in word_extract_lines])
    abstract=' '.join([line for line in abstract]) # modified
    
  #  article=SYB_RE.sub(r' ',article)
    print(article)
    article = bytes(article, 'utf-8')
    article=_DIGIT_RE.sub(b" 0",article) # modified
   # word_extract=SYB_RE.sub(r' ',word_extract)
    word_extract = bytes(word_extract, 'utf-8') # modified
    word_extract=_DIGIT_RE.sub(b" 0 ",word_extract)

    word_extract= (word_extract).decode("utf-8")  # modified
    article= (article).decode("utf-8") # modified
    extract=extract+ ' ' +word_extract
    art_tokens=article.split(' ')
    abs_tokens= abstract.split(' ')
    tokens = art_tokens + abs_tokens
    tokens = [t.strip() for t in tokens] # strip
    tokens = [t for t in tokens if t!=""] # remove empty
    vocab_counter.update(tokens)
    
  vocab_size=150000    
  print("Writing vocab file...")
  with open(os.path.join(finished_files_dir, "vocab"), 'w') as writer:
    for word, count in vocab_counter.most_common(vocab_size):
      writer.write(word + ' ' + str(count) + '\n')
  print("Finished writing vocab file")

  rake_object = rake.Rake("/home/mk/Downloads/swap-net-master/swap-net-summer/RAKE-tutorial-master/SmartStoplist.txt",1,1,1) # modified
  rake_keywords=rake_object.run(extract)
  keys=[k[0] for k in rake_keywords if not k[0].startswith('@entity')]
 
  keys=keys[:vocab_size]

  print("Writing keyword file...")
  with open(os.path.join(finished_files_dir, "keyword"), 'w') as writer:
    for word in keys:
#      if word in set(vocab_counter.elements()):
      writer.write(word  + '\n')
  print("Finished writing vocab file")


def get_emb_data(data_file,mode):
  print("Getting data for embeddings")
  complete_data=""
  data_list=[]
  data_list += read_text_file(data_file)

  with open(os.path.join(finished_files_dir, "embed_data_1.txt"), 'a') as writer:
    for idx,s in enumerate(data_list):
      print(cnn_tokenized_stories_dir)
      print(s)
      if os.path.isfile(os.path.join(cnn_tokenized_stories_dir, s)):
          story_file = os.path.join(cnn_tokenized_stories_dir, s)
          
      elif os.path.isfile(os.path.join(dm_tokenized_stories_dir, s)):
        story_file = os.path.join(dm_tokenized_stories_dir, s)

      #story_file = "/home/mk/Downloads/swap-net-master/swap-net-summer/dataset/finished_files/training_list.txt"
      #print(story_file)
      article_lines,abstract_lines,word_extract_lines=get_art_abs(story_file)

      

      article=' '.join([line for line in article_lines])
      #word_extract=' '.join([line for line in word_extract_lines])
      abstract=' '.join([line for line in abstract_lines])
      abstract=abstract.replace("<eos>","")
      complete_data=""
      complete_data+=article
      complete_data+=abstract

    
      writer.write(complete_data)


def get_emb_model():
  sent=gensim.models.word2vec.LineSentence(os.path.join(finished_files_dir, "embed_data_1.txt"))
  model=gensim.models.Word2Vec(sent,vector_size=100, window=6,min_count=2,negative=10)
  print("Training embed model")
  model.wv.save_word2vec_format(os.path.join(finished_files_dir, "embed_model.bin"),binary=True)


def write_to_bin(data_file, out_file,mode, makevocab=False):
  
  data_list = read_text_file(data_file)
 

  if makevocab:
    vocab_counter = collections.Counter()
  key=prepare_key_vocab()
  key=prepare_key()
  with open(out_file, 'wb') as writer:
    for idx,s in enumerate(data_list):
 
      if os.path.isfile(os.path.join(cnn_tokenized_stories_dir, s)):
        story_file = os.path.join(cnn_tokenized_stories_dir, s)
	
      elif os.path.isfile(os.path.join(dm_tokenized_stories_dir, s)):
        story_file = os.path.join(dm_tokenized_stories_dir, s)
	
      else:
        print("Error: Couldn't find tokenized story file", s ,"in either tokenized story directories ", cnn_tokenized_stories_dir ,"and ", dm_tokenized_stories_dir,". Was there an error during tokenization?")
        # Check again if tokenized stories directories contain correct number of files
        
      article_text,sent_label_text,word_text,word_label_text,abstract_text = get_everything(story_file,key)
      #print('ab text',abstract_text)
      # Write to tf.Example
      tf_example = example_pb2.Example()
      article_text = bytes(article_text, 'utf-8') # modified
      sent_label_text = bytes(sent_label_text, 'utf-8') # modified
      word_text = bytes(word_text, 'utf-8') # modified
      word_label_text = bytes(word_label_text, 'utf-8') # modified
      abstract_text = bytes(abstract_text, 'utf-8') # modified

      tf_example.features.feature['article_text'].bytes_list.value.extend([article_text])
      tf_example.features.feature['sent_label_text'].bytes_list.value.extend([sent_label_text])
      tf_example.features.feature['word_text'].bytes_list.value.extend([word_text])
      tf_example.features.feature['word_label_text'].bytes_list.value.extend([word_label_text])
      tf_example.features.feature['abstract_text'].bytes_list.value.extend([abstract_text])

      tf_example_str = tf_example.SerializeToString()
     # print(tf_example_str)
      str_len = len(tf_example_str)
      #print(str_len)
      writer.write(struct.pack('q', str_len))
      #writer.write(struct.pack('%ds' % str_len, tf_example_str))
      writer.write(struct.pack('%ds' % str_len, tf_example_str))
      # Write the vocab to file, if applicable
      if makevocab:
        art_tokens = article.split(' ')
        abs_tokens = abstract.split(' ')
        abs_tokens = [t for t in abs_tokens if t not in [SENTENCE_START, SENTENCE_END]] # remove these tags from vocab
        tokens = art_tokens + abs_tokens
        tokens = [t.strip() for t in tokens] # strip
        tokens = [t for t in tokens if t!=""] # remove empty
        vocab_counter.update(tokens)

  print("Finished writing file ", out_file,"\n")


def initialise():
  if not os.path.exists(cnn_tokenized_stories_dir): os.makedirs(cnn_tokenized_stories_dir)
  if not os.path.exists(dm_tokenized_stories_dir): os.makedirs(dm_tokenized_stories_dir)
  if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

  # Run stanford tokenizer on both stories dirs, outputting to tokenized stories directories
  for d in ['training','test','validation'] :
  
    tokenize_stories(os.path.join(cnn_summary_dir,d), cnn_tokenized_stories_dir)
    tokenize_stories(os.path.join(dm_summary_dir,d), dm_tokenized_stories_dir)

    file_list=os.listdir(os.path.join(cnn_summary_dir,d))
    file_list+=os.listdir(os.path.join(dm_summary_dir,d))
    #file_list=os.listdir(os.path.join(dm_summary_dir,d))
    with open(os.path.join(finished_files_dir, d + '_list.txt'), 'w') as writer:
      for file in file_list:
        writer.write(file +'\n')


  
if __name__ == '__main__':

  initialise()
  get_emb_data(os.path.join(finished_files_dir,'training_list.txt'),'training')
  get_emb_data(os.path.join(finished_files_dir,'validation_list.txt'),'validation')
  get_emb_data(os.path.join(finished_files_dir,'test.txt'),'test')
  get_emb_model()
  make_vocab(os.path.join(finished_files_dir,'training_list.txt'))
  write_to_bin(os.path.join(finished_files_dir,'training_list.txt'),os.path.join(finished_files_dir,'train.bin'),'training')
  write_to_bin(os.path.join(finished_files_dir,'test_list.txt'),os.path.join(finished_files_dir,'test.bin'),'test')
  write_to_bin(os.path.join(finished_files_dir,'validation_list.txt'),os.path.join(finished_files_dir,'val.bin'),'validation')
 # chunk_all()
