import cv2
import boto3
import time
import os
import sys
from botocore.exceptions import BotoCoreError, ClientError
from contextlib import closing
from tempfile import gettempdir
from pydub import AudioSegment
from pydub.playback import play

# Video stream
videoStreamUrl = "rtsp://10.2.10.2:9000/live"

# MP3 File
mp3File = "audio.mp3"

session = boto3.session.Session()
rekognition = session.client('rekognition')
polly = session.client('polly')


# Process data from response
def processData(response):
    print("Analysing data...")
    persons = []
    
    for person in response['Persons']:
        mask = False
        coverNose = False
        for bodyParts in person['BodyParts']:

            if bodyParts['Name'] == "FACE":
                if len(bodyParts['EquipmentDetections']) > 0:
                    for equipmentDetections in bodyParts['EquipmentDetections']:
                        if equipmentDetections['Type'] == "FACE_COVER":
                            mask = True
                            coverNose = equipmentDetections['CoversBodyPart']['Value']
                        
        persons.append({'id': person['Id'], 'mask': mask, 'coverNose': coverNose})

    del response

    print("Summarizing data...")
    num_persons_no_mask = 0
    num_persons_mask_wrong = 0
    num_persons_mask = 0
    try:
        for person in persons:
            if person['mask'] == True:
                num_persons_mask += 1
                if person['coverNose'] == False:
                    num_persons_mask_wrong += 1
            else:
                num_persons_no_mask += 1
            
    except Exception as e:
        print("Error processing data from request: " + str(e))
        return

    return {'num_persons_no_mask': num_persons_no_mask, 'num_persons_mask_wrong': num_persons_mask_wrong, 'num_persons_mask': num_persons_mask}


# Process frame from video stream
def processFrame(videoStreamUrl):
    print("Processing frame from video stream...")
    cap = cv2.VideoCapture(videoStreamUrl)
    ret, frame = cap.read()
    if ret:
        hasFrame, imageBytes = cv2.imencode(".jpg", frame)
        if hasFrame:
            
            response = rekognition.detect_protective_equipment(
                    Image={
                        'Bytes': imageBytes.tobytes(),
                    },
                    SummarizationAttributes={
                        'MinConfidence': 80,
                        'RequiredEquipmentTypes': ['FACE_COVER']
                    }
                )
    cap.release()
    print("Response: ", response)

    if len(response['Persons']) == 0:
        print("There are no persons in front of the camera")
        return

    
    
    data = processData(response)
    print("Data: ", data)
    if data == None:
        print("There is no data to process...")
        return
    if data['num_persons_no_mask'] > 0 or data['num_persons_mask_wrong'] > 0:
        if processTTS(data):
            playAudio()
    return


# Process Text to Speech from data
def processTTS(data):
    print("Processing TTS...")
    # Create text
    message_tts = "Atenção! Existe"

    if data['num_persons_no_mask'] > 0:
        message_tts += " " + str(data['num_persons_no_mask'])
        if data['num_persons_no_mask'] > 1:
            message_tts += " pessoas"
        else:
            message_tts += " pessoa"
        message_tts += " sem máscara."

        if data['num_persons_mask_wrong'] > 0:
            message_tts += " e"

    if data['num_persons_mask_wrong'] > 0:
        message_tts += " " + str(data['num_persons_mask_wrong'])
        if data['num_persons_mask_wrong'] > 1:
            message_tts += " pessoas"
        else:
            message_tts += " pessoa"
        message_tts += " com a máscara mal colocada."

    message_tts += " Para a segurança de todos, por favor coloquem a máscara e devidamente bem colocada. Obrigada!"
    print("TTS Message generated: " + message_tts)

    print("Getting audio from AWS Polly...")
    try:
        response = polly.synthesize_speech(
            Engine='standard',
            LanguageCode='pt-PT',
            OutputFormat='mp3',
            Text=message_tts,
            VoiceId='Ines'
        )
    except Exception as e:
        print("Error while generating audio: " + str(e))
        return False

    file = open(mp3File, 'wb')
    file.write(response['AudioStream'].read())
    file.close()
    return True

# Play audio
def playAudio():
    try:
        audio_stream = AudioSegment.from_mp3(mp3File)
        print("Playing audio...")
        play(audio_stream)
    except Exception as e:
        print("Could not play audio: " + str(e))


# Infinite cycle
print("Starting script...")
while (True):
    try:
        processFrame(videoStreamUrl)
    except Exception as e:
        print("Error: {}.".format(e))