�+  *�p=
�P@��K7�x�@2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch�'�&�?!2×~�R@)�'�&�?12×~�R@:Preprocessing2
HIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2�-�|���?!����A4@)6��Ϸ�?1س�]�V@:Preprocessing2�
QIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle �������?!��)e!,0@)�� @���?1�'mÊ0@:Preprocessing2�
uIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[4]::FiniteSkip::FlatMapC�+j�?!��K�E@)D�U��y�?1p�Sbj�@:Preprocessing2F
Iterator::Model��"�t5�?!�1��S@)uWv����?1VM;�@:Preprocessing2�
]Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave ~��7L�?!Y��h�?(@)Q�f��?1F�!l��@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism�.��?!����R@)`��9z�?1�ݒ��B�?:Preprocessing2v
?Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2Nё\�C�?!ҁ��g8�?)Nё\�C�?1ҁ��g8�?:Preprocessing2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[4]::FiniteSkip::FlatMap[0]::TFRecord��h㈵�?!���_���?)��h㈵�?1���_���?:Advanced file read2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[0]::FiniteSkip::FlatMap[0]::TFRecord�*5{��?!Y���5�?)�*5{��?1Y���5�?:Advanced file read2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[2]::FiniteSkip::FlatMap[0]::TFRecorda2U0*��?!��?=!�?)a2U0*��?1��?=!�?:Advanced file read2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[3]::FiniteSkip::FlatMap[0]::TFRecord�N�z1��?!�nė��?)�N�z1��?1�nė��?:Advanced file read2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[1]::FiniteSkip::FlatMap[0]::TFRecorde�I)���?!�qȭ�|�?)e�I)���?1�qȭ�|�?:Advanced file read2�
uIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[0]::FiniteSkip::FlatMapYLl>��?!C�b�i�?)/n���?1l��$x��?:Preprocessing2�
lIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[4]::FiniteSkip����u�?!G0���@)�vLݕ]�?1�I�.�?:Preprocessing2�
uIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[3]::FiniteSkip::FlatMapr�Z|
��?!�E��/#�?)�s����~?1�8Ю�p�?:Preprocessing2�
lIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[0]::FiniteSkip}"O����?!Ω+�_�?)�X�O0~?1/��p<��?:Preprocessing2�
uIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[2]::FiniteSkip::FlatMap�S:X��?!_*��-�?)!�> �M|?1�%_s�?:Preprocessing2�
uIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[1]::FiniteSkip::FlatMap���a��?!�ǟ�?v�?)O#-��#|?1�����?:Preprocessing2�
lIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[1]::FiniteSkip�+e�X�?!؏g�x��?)5A�} R{?1��^�c1�?:Preprocessing2�
lIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[3]::FiniteSkip0du�礗?!(nے��?)�~j�t�x?1��_����?:Preprocessing2�
lIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[2]::FiniteSkipZ�wg�?!u+ȟ\$�?)�*5{�x?1Y���5�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb��No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.Y      Y@q+Oy��,@"�
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"GPU(: B��No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.