## Examples

17. run_task_text_classification_ema_cola.sh
> 文本分类任务，增加Exponential Moving Average

16. task_relation_classification_semeval.py
> 关系分类任务，主要参考论文: Enriching Pre-trained Language Model with Entity Information for Relation Classification

15. task_sequence_labeling_ner_span.py
> 命名实体识别任务，采用MRC模式进行，主要参考论文: A Unified MRC Framework for Named Entity Recognition (实现方式有点不一样)

14. task_text_classification_alum_cola.py
> 文本分类任务，增加alum对抗，主要参考论文：Adversarial Training for Large Neural Language Models

13. task_text_classification_freelb_cola.py
> 文本分类任务，增加FreeLB对抗，主要参考论文: FreeLB: Enhanced Adversarial Training for Language Understanding

12. task_text_classification_sda_cola.py
> 文本分类任务，增加self-distillation，主要参考论文： Improving BERT Fine-Tuning via Self-Ensemble and Self-Distillation

11. task_text_classification_fgm_cola.py
> 文本分类任务，增加FGM对抗

10. task_attribute_value_extract_crf.py
> 实体属性提取任务，主要参考论文: Scaling Up Open Tagging from Tens to Thousands: Comprehension Empowered Attribute Value Extraction from Product Title

9. task_text_classification_lookahead_cola.py
> 文本分类任务，增加Lookahead优化器，主要参考论文：Lookahead Optimizer: k steps forward, 1 step back

8. task_text_classification_mdp_cola.py
> 文本分类任务，增加multisample-dropout，主要参考论文：Multi-Sample Dropout for Accelerated Training and Better Generalization

7. task_multilabel_classification_toxic.py
> 多标签分类任务，数据来源与Kaagle中的Toxic评论比赛

6. task_triple_similarity_epidemic.py
> Triple文本相似任务，主要采用Triple Network形式

5. task_sentence_similarity_lcqmc.py
> 句子相似任务，形式为[CLS]sentence_A[SEP]sentence_B[SEP]

4. task_sequence_labeling_ner_crf.py
> 命名实体识别任务，使用CRF解码器

3. task_sequence_labeling_layer_lr_ner_crf.py
> 命名实体识别任务，调整CRF层学习率大小

2. task_siamese_similarity_afqmc.py
> 句子相似任务，采用siamese network结构

1. task_text_classification_cola.py
> 文本分类任务，主要形式为[CLS]sentence[SEP]
