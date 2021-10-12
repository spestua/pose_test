import mediapipe as mp
import cv2
import numpy as np
import collections
import streamlit as st
import tempfile
import pandas as pd

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

threshold = 0.1
heightAngle = [350, 370, 390, 410]
dictAngle = {'leftHand': [15, 13, 11],
             'leftLeg': [23, 25, 27],
             'rightHand': [16, 14, 12],
             'rightLeg': [24, 26, 28]}

dir_temp = tempfile.TemporaryDirectory()

st.set_page_config(
     page_title="Pose Estimation",
     page_icon="🎾",
)

st.title('🎾 Pose Estimation 🎾')


complexity = st.select_slider("Укажите сложность модели", options=[0, 1, 2], value=1)
precision = st.select_slider("Укажите частоту обработки", options=[1, 2, 3, 4, 5], value=2)
uploaded_file = st.file_uploader("", type='mp4')
tfile = tempfile.NamedTemporaryFile(delete=False)

if uploaded_file is not None:
    video = uploaded_file.read()
    tfile.write(video)
    
    st.video(video)
    with st.spinner('Обработка...'):
        cap = cv2.VideoCapture(tfile.name)

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(5))
        frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        temp_name = next(tempfile._get_candidate_names())
        out = cv2.VideoWriter(f'{dir_temp.name}/output.mp4', fourcc, fps, (frame_width, frame_height))
        output_data_raw = []

        with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=complexity) as pose:
            bar = st.progress(0)
            for frame_ix in (range(frame_num)):
                bar.progress((frame_ix+1)/frame_num)

                ret, frame = cap.read()
                if not ret: break
                if frame_ix % precision == 0:
                    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                else:
                    pass
                if not results.pose_landmarks: continue
                
                frameWidth = frame.shape[1]
                frameHeight = frame.shape[0]

                dictAngle = collections.OrderedDict(sorted(dictAngle.items()))

                angles_temp = {}
                for j, i in enumerate(dictAngle):

                    p1, p2, p3 = [np.array([i.x*frameWidth, i.y*frameHeight]) for i in np.array(results.pose_landmarks.landmark)[dictAngle[i]]]

                    if str(p1) != 'None' and str(p2) != 'None' and str(p3) != 'None':

                        ba = p1 - p2
                        bc = p3 - p2

                        p2_int = [int(p2[0]), int(p2[1])]
                        if (i == 'leftHand') or (i == 'leftLeg'): pointAngle = (p2_int[0] + 15, p2_int[1] + 18)
                        if (i == 'rightHand') or (i == 'rightLeg'): pointAngle = (p2_int[0] - 50, p2_int[1])

                        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                        angle = np.arccos(cosine_angle)
                        angle_deg = int(np.degrees(angle))
                        cv2.putText(frame, str(angle_deg), pointAngle, cv2.FONT_ITALIC, 0.4, (255,255,255), 2, cv2.LINE_AA)
                        cv2.putText(frame, str(i + ": "), (15, heightAngle[j]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1,
                                    cv2.LINE_AA)
                        cv2.putText(frame, str(angle_deg), (90, heightAngle[j]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
                    angles_temp[i] = angle_deg
                angles_temp.update({'Time': cap.get(cv2.CAP_PROP_POS_MSEC)})
                output_data_raw.append(angles_temp)

                    # Draw pose landmarks.
                annotated_image = frame.copy()
                mp_drawing.draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                out.write(annotated_image)
            cap.release()
            out.release()
            cv2.destroyAllWindows()

            output_video = open(f'{dir_temp.name}/output.mp4', 'rb').read()
            output_data = pd.DataFrame(output_data_raw).rename(columns={'leftHand': 'Левая рука',
                                                                        'leftLeg': 'Левая нога',
                                                                        'rightHand': 'Правая рука',
                                                                        'rightLeg': 'Правая нога',
                                                                        'Time': 'Время (мс)'})
                                                                        
            output_data.drop_duplicates(subset=['Левая рука','Левая нога', 'Правая рука', 'Правая нога'],
                                        inplace=True)

            output_data.to_excel(f'{dir_temp.name}/output.xlsx')

            output_excel = open(f'{dir_temp.name}/output.xlsx', 'rb').read()


            st.video(output_video)
            st.dataframe(output_data)
            st.line_chart(output_data.set_index('Время (мс)')[['Левая рука','Левая нога', 'Правая рука', 'Правая нога']])
            st.download_button(label="Скачать данные", data=output_excel, file_name='output.xlsx')

dir_temp.cleanup()
