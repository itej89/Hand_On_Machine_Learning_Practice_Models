  *��n�(Q@0�$��v@2�
QIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle o�e����?!�4���P@)�,^,�?1H��.@:Preprocessing2
HIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2����D�?!e�|���S@)NB�!�?10�C���*@:Preprocessing2�
]Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave ���ډ�?!ld/��I@)a3�ٲ�?1ޞ��Ӳ)@:Preprocessing2v
?Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2F]k�SU�?!Y�7r@@)F]k�SU�?1Y�7r@@:Preprocessing2�
uIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[4]::FiniteSkip::FlatMap:���u�?!����(@)��I`s�?1�>��@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism��[���?!(Ҷ]@'@)���;�?1f��XA@:Preprocessing2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[4]::FiniteSkip::FlatMap[0]::TFRecord��Sݘ?!�b߹C@)��Sݘ?1�b߹C@:Advanced file read2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchbg
�ט?!E>���>@)bg
�ט?1E>���>@:Preprocessing2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[0]::FiniteSkip::FlatMap[0]::TFRecordR�d=��?!&,<~�@)R�d=��?1&,<~�@:Advanced file read2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[1]::FiniteSkip::FlatMap[0]::TFRecordԷ�鲘�?!���xu@)Է�鲘�?1���xu@:Advanced file read2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[3]::FiniteSkip::FlatMap[0]::TFRecord.�_x%Ƀ?!��xI��@).�_x%Ƀ?1��xI��@:Advanced file read2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[2]::FiniteSkip::FlatMap[0]::TFRecord���"��?!Q�V\ʭ@)���"��?1Q�V\ʭ@:Advanced file read2�
uIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[0]::FiniteSkip::FlatMap�>#K�?!#}Z:��@)7����?1 �8� @:Preprocessing2�
lIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[4]::FiniteSkip*����ί?!.^_�a{,@)����4c�?1k���#�?:Preprocessing2�
lIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[0]::FiniteSkip���
؞?!�M�Л�@)�o_��?1�B�XR��?:Preprocessing2�
uIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[2]::FiniteSkip::FlatMap�n/i�֑?!�`��X�@)u����?1�ha4��?:Preprocessing2�
uIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[1]::FiniteSkip::FlatMap���=�>�?!_��*� @)�x#��?1��ֹUv�?:Preprocessing2�
lIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[1]::FiniteSkipy]�`7l�?!�r�|W�@)-[닄�|?1��F��?:Preprocessing2�
lIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[2]::FiniteSkip�(yu��?!%%��Zd@)y�&1�|?1��I����?:Preprocessing2�
uIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[3]::FiniteSkip::FlatMapH�V
�?!z��P��@)�{�|?1�_���?:Preprocessing2�
lIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[3]::FiniteSkip��)�?!�C7=�@)��V�I�{?1�k:��?:Preprocessing2F
Iterator::Model�;���ܬ?!�ɏ���)@)ap��/w?1���t��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb��No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.