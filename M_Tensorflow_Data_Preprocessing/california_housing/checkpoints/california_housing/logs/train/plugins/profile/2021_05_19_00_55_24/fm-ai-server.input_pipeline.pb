  *}?5^��R@�l����~@)      0=2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch���?!dB�C�6@)���?1dB�C�6@:Preprocessing2�
QIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle ���{��?!H١�eL@)	��g��?1�t+�ii0@:Preprocessing2
HIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2�&7���?!��;m�P@)�A�۽ܯ?16�-�%@:Preprocessing2�
]Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave ��C�l�?!��0D@)P��ôo�?1�}�u��@:Preprocessing2�
uIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[4]::FiniteSkip::FlatMap=��S��?!�<I#@)��L���?1��3�@:Preprocessing2v
?Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2_A��h:�?!2>��@)_A��h:�?12>��@:Preprocessing2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[4]::FiniteSkip::FlatMap[0]::TFRecord�EB[Υ�?!�j�P��@)�EB[Υ�?1�j�P��@:Advanced file read2]
&Iterator::Model::MaxIntraOpParallelismD�R�Z�?!O����:@)!��=@��?1N?�?a%@:Preprocessing2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[0]::FiniteSkip::FlatMap[0]::TFRecord�^`V(ҍ?!"^�X�v@)�^`V(ҍ?1"^�X�v@:Advanced file read2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[1]::FiniteSkip::FlatMap[0]::TFRecord��R^+��?!������@)��R^+��?1������@:Advanced file read2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[2]::FiniteSkip::FlatMap[0]::TFRecordG�ŧ �?!�=�zx @)G�ŧ �?1�=�zx @:Advanced file read2�
lIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[4]::FiniteSkip�fe����?!� W�L�&@)o/i��Q�?13��]B�?:Preprocessing2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[3]::FiniteSkip::FlatMap[0]::TFRecord����h�?!�Ϟ�i��?)����h�?1�Ϟ�i��?:Advanced file read2�
uIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[0]::FiniteSkip::FlatMap�J�({K�?!�S~[@)6���Ą?13�Яπ�?:Preprocessing2�
lIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[0]::FiniteSkip���k��?!<��c�J@)f1���6�?1xD.ѽ�?:Preprocessing2�
lIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[2]::FiniteSkip�s���?!d��3�@)�P�y�?1b+�76Z�?:Preprocessing2�
lIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[1]::FiniteSkipgE�D��?!�v���@)L��1%�?1c�(	��?:Preprocessing2�
uIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[1]::FiniteSkip::FlatMaplMK���?!�X�r�@)*oG8-x�?1��w����?:Preprocessing2�
uIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[2]::FiniteSkip::FlatMap#��~j��?!f��wLu@)*oG8-x�?1��w����?:Preprocessing2�
uIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[3]::FiniteSkip::FlatMap��4�?!~����*
@)i��Q��?1	��iW[�?:Preprocessing2�
lIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[3]::FiniteSkip�	ܺ���?!t���`J@)`���Y~?1��d���?:Preprocessing2F
Iterator::ModelwJ���?!G5�σ�;@)tF��_x?1��V�<��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb��No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.