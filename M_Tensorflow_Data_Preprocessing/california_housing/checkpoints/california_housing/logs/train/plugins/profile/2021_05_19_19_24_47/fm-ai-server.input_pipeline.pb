  *��/ݬS@�Zdgx@2�
QIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle ���^D��?!H�±S�P@)x` �C��?1���}�2@:Preprocessing2
HIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2�o��1=�?!^��4LYS@)/o�j�?1���ć&@:Preprocessing2�
]Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave )��q�?!�X���G@)�TPQ�+�?1���맠!@:Preprocessing2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[4]::FiniteSkip::FlatMap[0]::TFRecord+��<��?!A54ð@)+��<��?1A54ð@:Advanced file read2]
&Iterator::Model::MaxIntraOpParallelism��O�?!f�V,@)�Y-��D�?1�V��k@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch<g���?!�Bͳy@@)<g���?1�Bͳy@@:Preprocessing2v
?Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2Uka�9�?!S�=YcU@)Uka�9�?1S�=YcU@:Preprocessing2�
uIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[4]::FiniteSkip::FlatMap�B��˰?!Ӟ��)�+@)Ov3��?1e�ݏ?@:Preprocessing2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[0]::FiniteSkip::FlatMap[0]::TFRecord�A�L��?!��_]�p@)�A�L��?1��_]�p@:Advanced file read2�
uIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[0]::FiniteSkip::FlatMap!��F��?!�2b��@)f��@�9�?1Krd��V@:Preprocessing2F
Iterator::Model^,��׳?!4	ZVv�0@)#-��#��?1��4�@:Preprocessing2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[1]::FiniteSkip::FlatMap[0]::TFRecordz����?!��Z�p @)z����?1��Z�p @:Advanced file read2�
uIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[3]::FiniteSkip::FlatMapB'����?!���F@@)�b*���?1�1��}@ @:Preprocessing2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[3]::FiniteSkip::FlatMap[0]::TFRecordj���v��?!NG�$@ @)j���v��?1NG�$@ @:Advanced file read2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[2]::FiniteSkip::FlatMap[0]::TFRecordHP�sׂ?!���`�?)HP�sׂ?1���`�?:Advanced file read2�
lIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[0]::FiniteSkip幾	�?!�q�#�]@)O�9����?1s�@Rx��?:Preprocessing2�
uIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[1]::FiniteSkip::FlatMap�(���ǒ?!�p5J@F@)�5�;Nс?1��{��?:Preprocessing2�
lIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[1]::FiniteSkip/R(__�?!+U��U�@);S��.�?1Vs�ל�?:Preprocessing2�
lIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[4]::FiniteSkip(�쿲?!N@���8/@)��~��@?1��S�?:Preprocessing2�
uIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[2]::FiniteSkip::FlatMap�6�ُ�?!y��0q@)O;�5Y�~?1=�^��?:Preprocessing2�
lIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[3]::FiniteSkipG��t��?!���4cd@)Է��x?19���t��?:Preprocessing2�
lIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[2]::FiniteSkip�Hh˹�?!\ˇ�7@)G�ŧ x?1~�7�	��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb��No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.