# omr-web-app
## Playback
Web-based Optical Music Recognition tool that translates musical notes on monophonic scores to playable MIDI files.
This project was created as a part of Final Year Project with  reference to [Web-OMR](https://github.com/liuhh02/web-omr).
You can see the full article explaining that project [here](https://towardsdatascience.com/i-built-a-music-sheet-transcriber-heres-how-74708fe7c04c). 

## Getting Started
This web app is developed with Flask on a [tensorflow model](https://github.com/calvozaragoza/tf-deep-omr) built by Calvo-Zaragoza et al. published as [End-to-End Neural Optical Music Recognition of Monophonic Scores](https://www.mdpi.com/2076-3417/8/4/606) in the Applied Sciences Journal 2018.

To get started, follow the steps below:

 1. Install the following dependencies: tensorflow v1, flask, opencv, pygame
 2. Download the [semantic model](https://grfia.dlsi.ua.es/primus/models/PrIMuS/Semantic-Model.zip) developed by Calvo-Zaragoza et al.
 3. Download the [Semantic to MIDI Converter](https://grfia.dlsi.ua.es/primus/primus_converter.tgz)

If you would like to train the semantic model yourself, head over to the tensorflow model [Github repository](https://github.com/calvozaragoza/tf-deep-omr) for instructions and download the [PrIMuS dataset](https://grfia.dlsi.ua.es/primus/).

## Folder Structure
Make sure your folder structure is as follows:
```
app.py
vocabulary_semantic.txt

├── Semantic-Model
|   ├── semantic_model.meta
|   ├── semantic_model.index
|   └── semantic_model.data-00000-of-00001
├── primus_conversor
|   └── omr-3.0-SNAPSHOT.jar
├── templates
|   ├── index.html
|   └── result.html
├── static
|   └── css
|        └── bulma.min.css
```
## Run the Web App!
Once everything has been set up as above, head over to your terminal / command prompt. Change the directory to the directory with your `app.py` file and run `python app.py`. Wait for a few seconds and you should receive a message on the link you should go to in order to view the web app. Go to the URL, upload your music sheet and get the result! 

The converted semantic file and MIDI file will be saved to server_files as `output.semantic` and `output.mid` respectively. The MIDI file can be played on the user side.

## Acknowledgements
A huge thanks to Calvo-Zaragoza et al. for building this awesome deep learning model, and for sharing the trained model, dataset and code.
And a huge thanks to Liu Haohui for the basic structure of the web app which is used here.
