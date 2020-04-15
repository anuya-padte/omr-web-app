from flask import Flask,request,jsonify,send_from_directory,render_template, send_file, make_response
from functools import wraps, update_wrapper
from werkzeug.utils import secure_filename
import os
import pygame
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import cv2
import numpy as np
import subprocess
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
app = Flask(__name__, static_url_path='')

def sparse_tensor_to_strs(sparse_tensor):
    indices= sparse_tensor[0][0]
    values = sparse_tensor[0][1]
    dense_shape = sparse_tensor[0][2]

    strs = [ [] for i in range(dense_shape[0]) ]

    string = []
    ptr = 0
    b = 0

    for idx in range(len(indices)):
        if indices[idx][0] != b:
            strs[b] = string
            string = []
            b = indices[idx][0]

        string.append(values[ptr])

        ptr = ptr + 1

    strs[b] = string

    return strs


def normalize(image):
    return (255. - image)/255.


def resize(image, height):
    width = int(float(height * image.shape[1]) / image.shape[0])
    sample_img = cv2.resize(image, (width, height))
    return sample_img

voc_file = "vocabulary_semantic.txt"
model = "Semantic-Model/semantic_model.meta"

tf.reset_default_graph()
sess = tf.InteractiveSession()
# Read the dictionary
dict_file = open(voc_file,'r')
dict_list = dict_file.read().splitlines()
int2word = dict()
for word in dict_list:
    word_idx = len(int2word)
    int2word[word_idx] = word
dict_file.close()

# Create a directory in a known location to save files to.
output_dir = os.path.join('server_files')

# Restore weights
saver = tf.train.import_meta_graph(model)
saver.restore(sess,model[:-5])

graph = tf.get_default_graph()

input = graph.get_tensor_by_name("model_input:0")
seq_len = graph.get_tensor_by_name("seq_lengths:0")
rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
height_tensor = graph.get_tensor_by_name("input_height:0")
width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
logits = tf.get_collection("logits")[0]

# Constants that are saved inside the model itself
WIDTH_REDUCTION, HEIGHT = sess.run([width_reduction_tensor, height_tensor])

decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len)

@app.route('/server_files/<filename>')
def send_img(filename):
    return send_from_directory(output_dir, filename)

@app.route('/play_midi_file')
def play_midi_file():
    pygame.mixer.init()
    freq, size, chan = pygame.mixer.get_init()
    BUFFER = 3072
    filename = output_dir+"/output.mid"
    pygame.mixer.init(freq, size, chan, BUFFER)

    pygame.init()
    pygame.mixer.init()
    clock = pygame.time.Clock()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    print("play song")
    while pygame.mixer.music.get_busy():
        clock.tick(1000)

    return "nothing"

@app.route("/")
def root():
    return render_template('index.html')

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        img = f
        image = Image.open(img).convert('L')
        image = np.array(image)
        image = resize(image, HEIGHT)
        image = normalize(image)
        image = np.asarray(image).reshape(1,image.shape[0],image.shape[1],1)

        seq_lengths = [ image.shape[2] / WIDTH_REDUCTION ]
        prediction = sess.run(decoded,
                            feed_dict={
                                input: image,
                                seq_len: seq_lengths,
                                rnn_keep_prob: 1.0,
                            })
        str_predictions = sparse_tensor_to_strs(prediction)

        array_of_notes = []

        for w in str_predictions[0]:
            array_of_notes.append(int2word[w])

        # name_of_file = f.filename.split('.')[0]
        # semantic_file_path = output_dir+"/semantic/"+name_of_file+".semantic"
        # midi_file_path = output_dir+"/midi/"+name_of_file+".mid"
        name_of_file = f.filename

        semantic_file_path = output_dir+"/output.semantic"
        midi_file_path = output_dir+"/output.mid"
        file2 = open(semantic_file_path,"w")
        for word in array_of_notes:
            file2.write(word+"\t")
        file2.close()
        # img.save(output_dir+"/output.png")
        subprocess.call(['java', '-cp', 'primus_conversor/omr-3.0-SNAPSHOT.jar', 'es.ua.dlsi.im3.omr.encoding.semantic.SemanticImporter', semantic_file_path, midi_file_path])
    return render_template('result.html', file_name = name_of_file)

if __name__=="__main__":
    app.run()
