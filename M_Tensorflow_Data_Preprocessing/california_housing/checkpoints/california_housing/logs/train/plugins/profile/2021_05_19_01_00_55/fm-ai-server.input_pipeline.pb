  *�$��kT@%��#�@2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchbg
��8@!MԙЩ"V@)bg
��8@1MԙЩ"V@:Preprocessing2�
QIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle 3�68��?!D��!@)��i����?1]`�vxS@:Preprocessing2
HIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2�۽�'G�?!���S$@)���Z(�?1	��T�?:Preprocessing2�
]Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave @/ܹ0��?!�W�R��@)���+�z�?1�r.+�0�?:Preprocessing2�
uIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[4]::FiniteSkip::FlatMap�6���N�?!T#�;��?)0H�����?1^�D���?:Preprocessing2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[4]::FiniteSkip::FlatMap[0]::TFRecordMK��F>�?!J�e�&�?)MK��F>�?1J�e�&�?:Advanced file read2v
?Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2eM�?!r_
2�{�?)eM�?1r_
2�{�?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismo�[tr@!؄E1�KV@)��t_Μ?1�E�U0l�?:Preprocessing2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[0]::FiniteSkip::FlatMap[0]::TFRecord����j�?!sf��E�?)����j�?1sf��E�?:Advanced file read2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[2]::FiniteSkip::FlatMap[0]::TFRecord,����?!�s�N��?),����?1�s�N��?:Advanced file read2�
lIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[4]::FiniteSkip�M�M�g�?!*��?)��@��ǈ?1${SE���?:Preprocessing2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[1]::FiniteSkip::FlatMap[0]::TFRecordE�a���?!�׍ e�?)E�a���?1�׍ e�?:Advanced file read2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[3]::FiniteSkip::FlatMap[0]::TFRecord��^�2�?!U���?)��^�2�?1U���?:Advanced file read2�
uIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[0]::FiniteSkip::FlatMap�ypw�n�?!B<mA�r�?)HP�s�?1st����?:Preprocessing2�
lIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[2]::FiniteSkip�S:X��?!�_���?)^K�=��?1�����?:Preprocessing2�
lIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[0]::FiniteSkip��im�?!��@N�?)�^���?1 M��_��?:Preprocessing2�
uIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[1]::FiniteSkip::FlatMap撪�&��?!�-��H�?)��yUg�?1���$VY�?:Preprocessing2�
lIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[1]::FiniteSkipd��3�Ġ?!W�����?)�/��"�?1�mIG���?:Preprocessing2�
uIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[2]::FiniteSkip::FlatMapÁ�,`�?!N�ZP�?)YLl>��?1����6��?:Preprocessing2�
lIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[3]::FiniteSkip��(#. �?!�h'���?)m����?1ѵ����?:Preprocessing2�
uIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Interleave[3]::FiniteSkip::FlatMap�je�/��?!(v��}L�?)~�k�,	�?1���޼�?:Preprocessing2F
Iterator::ModelCus�}@!m6�~SV@)�I+�v?1AT�fw�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb��No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.