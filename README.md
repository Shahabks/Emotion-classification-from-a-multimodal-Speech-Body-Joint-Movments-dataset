# Emotion-classification-from-a-multimodel-Speech-Body-Joint-Movments-dataset
A preliminary experiment setup for Independent-language mental status - classification 

***Data***

We created an Audio/Body-Joint-Movement signals Dataset and labelled according to emotional status of the observation subjects. This data was fed into an experiment to demonstrate the process of the data-processing and machine learning models building. The dataset contains about 1300 files and eaxh file was rated emotional validity, intensity, and genuineness by the domain-experts.

Each file was labelled by 14 digits which helps us identify and trace back the characteristics: modality (01 = body-movement, 02 = audio), emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised), emotional intensity (01 = normal, 02 = strong), statement (01 = “Scenario-1”, 02 = “Scenarios-2”), repetition (01 = 1st repetition, 02 = 2nd repetition), data-collection-type (01=male, 02=female, 03=English-as 1st language, 04=other-language, 05=collecting mode).

***Experimet***

For this experiment, we have extracted 28 features from the audio from each file and 9 features from the body-joint-movement signals that I shall explain further down.  

***Theory and Rationale***

We will offer a multimodel approach: - Audio-Visual features analysis for mental status classification. Research and the panel of experts of our consortium accept and consider this approach. Multimodal observations combinations have complementary effects and from a data science viewpoint, this approach will increase machine learning model accuracy [A]. The challenge in multimodal feature learning is how and at what stage to fuse data from multiple modalities. This challenge is complicated by the high dimensionality of raw data, differing temporal resolutions, and differing temporal dynamics across modalities.Surveys on the general problem of sensor fusion [B] and speciﬁcally on fusion for affect recognition [C], [D] are available. Fusion can be achieved at early model stages close to the raw sensor data, or at a later stage by combining independent models. In early or feature-level fusion, features are extracted independently and then concatenated for further learning of a joint feature representation; this allows the model to capture correlations between the modalities. Late or decision-level fusion aggregates the results of independent recognition models. The literature generally reports that decision-level fusion works better for affect recognition given the datasets and models currently used [E]. While decision-level fusion typically only involves simple score weighing, feature-level fusion is a representation learning task that may beneﬁt from deep learning. 

In Audio-Visual observations combination there are two common approaches; in approach-1 reseachers took facial expressions while in approach-2 researchers took body-part-movement expressions with speech to study mental-status. In the feature-level fusion with neural networks approach, joint feature representations are learned without considering the temporal context for fusion. For both modalities, body-part-movement features are extracted using FER and SER methods that may involve both handcrafted and deep features (see Sections 3.1 and 3.2). A fully connected DNN, typically initialized via unsupervised pretraining, then learns a high-level joint feature representation of both modalities as an improvement over “shallow” feature fusion. Kim et al. [F] demonstrated how this can be achieved with DBNs. This approach is feasible especially in cases where the goal is to label each body-movement with one affective state. Alternatively, joint feature representations can be learned at the frame level, and then aggregated to the together. [G] used a DBN to fuse frame-level audio-visual features learned independently via CNNs; the learned features are average pooled for classiﬁcation at the body-movemet level and lead to an improvement over state-of-the-art methods. Feature-level fusion with RNNs (J2 in Fig. 6). Especially when predictions are required at the frame level for dimensional affective states, feature-level fusion could beneﬁt by taking into account the temporal context. Modeling via RNNs makes this possible, potentially improving model robustness and helping to deal with temporal lags between modalities [H]

This preliminary experiment reported that dynamic feature fusion can lead to performance improvements compared to simpler fusion strategies. However, several other studies based on handcrafted features found that decision-level fusion on top of individual LSTM models leads to better performance.

Learning from raw audio-visual data with two CNNs, Tzirakis et al. [I] used a two-layer LSTM network for feature fusion, which was found to outperform the state of the art.

##### [A] ***Z. Zeng, M. Pantic, G. I. Roisman, and T. S. Huang, “A survey of affect recognition methods: Audio, visual, and spontaneous expressions,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 31, no. 1, pp. 39–58, 2009.***
##### [B] ***F.Lingenfelser,J.Wagner,andE.Andr´e,“Asystematicdiscussion of fusion techniques for multi-modal affect recognition tasks,” in Proc. Int. Conf. Multimodal Interact., 2011, pp. 19–26.***
##### [C] ***S. K. D’Mello and J. Kory, “A review and meta-analysis of multimodal affect detection systems,” ACM Comput. Surveys, vol. 47, no. 3, p. 43, 2015.***
##### [D} ***S. Poria, E. Cambria, R. Bajpai, and A. Hussain, “A review of affective computing: From unimodal analysis to multimodal fusion,” Inf. Fusion, vol. 37, pp. 98–125, 2017***
##### [E] ***F.Ringeval,F.Eyben,E.Kroupi,A.Yuce,J.-P.Thiran,T.Ebrahimi, D. Lalanne, and B. Schuller, “Prediction of asynchronous dimensional emotion ratings from audio-visual and physiological data,” Pattern Recognit. Lett., vol. 66, pp. 22–30, 2015***
##### [F] ***H. Ranganathan, S. Chakraborty, and S. Panchanathan, “Multimodal emotion recognition using deep learning architectures,” in Proc. Winter Conf. Appl. Comput. Vision, 2016, pp. 1–9***
##### [G} ***S. Zhang, S. Zhang, T. Huang, W. Gao, and Q. Tian, “Learning affective features with a hybrid deep model for audio-visual emotion recognition,” IEEE Trans. Circuits Syst. Video Technol., 2017***
##### [H] ***F. Lingenfelser, J. Wagner, J. Deng, R. Bruckner, B. Schuller, and E. Andre, “Asynchronous and event-based fusion systems for affect recognition on naturalistic data in comparison to conventional approaches,” IEEE Trans. Affect. Comput., 2016***
##### [I} ***P. Tzirakis, G. Trigeorgis, M. A. Nicolaou, B. Schuller, and S. Zafeiriou, “End-to-end multimodal emotion recognition using deep neural networks,” IEEE J. Sel. Top. Signal Process., vol. 11, no. 8, pp. 1301–1309, 2017.***

![alt text](https://drive.google.com/uc?id=1q0hGl0lcM5JuUskS1VaXvSAk3vOn4giV)

# Analysis

We are using Colab, a Google Cloud environment for jupyter, so we need to import our files from Google Drive and then install LibROSA, a python package for music and audio analysis.

After the import, we will plot the signal of the first file.
