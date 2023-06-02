#!/bin/bash
clear
#set-up for single machine or cluster based execution
. ./cmd.sh
#set the paths to binaries and other executables
[ -f path.sh ] && . ./path.sh
train_cmd=run.pl
decode_cmd=run.pl

numLeavesMLLT=2000
numGaussMLLT=16000
numLeavesSAT=2000
numGaussSAT=16000

train_nj=11
decode_nj=5


#================================================
#	SET SWITCHES
#================================================

data_prep_sw=0

MFCC_extract_sw=1

mono_train_sw=1
mono_test_sw=1

tri1_train_sw=1
tri1_test_sw=1

tri2_train_sw=1
tri2_test_sw=1

tri3_train_sw=1
tri3_test_sw=1

tdnn_train_sw=0
tdnn_test_sw=0

dnn_train_sw=1
dnn_test_sw=1

#================================================
# Set Directories
#================================================

feat=dys_comb_vad

train_data=train_dys

test_data1=test_dys
test_data2=dev_dys

train_dir=data_${feat}/$train_data

test_dir1=data_${feat}/${test_data1}
test_dir2=data_${feat}/${test_data2}

graph_dir=graph_${test_data1}

decode_dir1=decode_${test_data1}
decode_dir2=decode_${test_data2}

vad_suffix=_pow_no_sil

#================================================
# Set LM Directories
#================================================
n_gram=2

#lm_suffix=

lm_suffix=_uas

train_lang=lang${lm_suffix}

train_lang_dir=data_${feat}/$train_lang

test_lang_dir=data_${feat}/lang${lm_suffix}_test

lm_arpa=${n_gram}-gram_lm_uas

libri_lm_dir=data_${feat}/local/LM_Resources

lm_arpa_dir=data_${feat}/local/lm${lm_suffix}

tmp_lm_arpa_dir=data_${feat}/local/tmp_lm${lm_suffix}


#====================================================

if [ $data_prep_sw == 1 ]; then

echo ============================================================================
echo "         Data & Lexicon & Language Model Preparation        "
echo ============================================================================

local_timit/libri_prepare_dict_bg.sh --n_gram $n_gram --srcdir data_${feat}/local/data --dir $dict_dir \
 --lmdir $lm_arpa_dir --tmpdir $tmp_lm_arpa_dir --language_model $lm_arpa

# Caution below: we remove optional silence by setting "--sil-prob 0.0",
# in TIMIT the silence appears also as a word in the dictionary and is scored.
utils/prepare_lang.sh --sil-prob 0.0 --position-dependent-phones false --num-sil-states 3 \
 $dict_dir "SIL" $tmp_langdir $train_lang_dir
 
utils/format_lm.sh $train_lang_dir $lm_arpa_dir/${lm_arpa}.arpa.gz $dict_dir/lexicon.txt $test_lang_dir

fi


if [ $MFCC_extract_sw == 1 ]; then

echo ============================================================================
echo "         MFCC Feature Extration & CMVN + Validation        "
echo ============================================================================

#extract MFCC features and perfrom CMVN

mfccdir=${feat}_feats


for datadir in $test_data1  $train_data $test_data2 ; do # 
	
	utils/fix_data_dir.sh data_${feat}/${datadir}
	utils/validate_data_dir.sh data_${feat}/${datadir}
	utils/copy_data_dir.sh data_${feat}/$datadir data_${feat}/${datadir}_pow || exit 1;
	utils/copy_data_dir.sh data_${feat}/$datadir data_${feat}/${datadir}_ener || exit 1;

	steps/make_mfcc.sh --mfcc-config conf/mfcc_pow.conf --cmd "$train_cmd" --nj 10 \
	 data_${feat}/${datadir}_pow exp_${feat}/make_mfcc/${datadir}_pow \
	 $mfccdir/${datadir}_pow || exit 1;
	 
	steps/make_mfcc.sh --mfcc-config conf/mfcc_ener.conf --cmd "$train_cmd" --nj 10 \
	 data_${feat}/${datadir}_ener exp_${feat}/make_mfcc/${datadir}_ener \
	 $mfccdir/${datadir}_ener || exit 1;
	  
	sid/compute_vad_decision.sh --nj 5 --cmd "$train_cmd" \
    	 data_${feat}/${datadir}_ener exp_${feat}/make_mfcc/${datadir}_ener $mfccdir/${datadir}_ener || exit 1;
    	 
        cp data_${feat}/${datadir}_ener/vad.scp data_${feat}/${datadir}_pow || exit 1;
        utils/fix_data_dir.sh data_${feat}/${datadir}_pow || exit 1;
        
  	local/nnet3/xvector/prepare_feats_for_egs.sh --nj 5 --cmd "$train_cmd" \
    	  data_${feat}/${datadir}_pow data_${feat}/${datadir}_pow_no_sil $mfccdir/${datadir}_pow_no_sil || exit 1;
    	  
    	cp data_${feat}/${datadir}/text data_${feat}/${datadir}_pow_no_sil || exit 1;
  	utils/fix_data_dir.sh data_${feat}/${datadir}_pow_no_sil || exit 1;

 	steps/compute_cmvn_stats.sh data_${feat}/${datadir}_pow_no_sil \
 	 exp_${feat}/make_mfcc/${datadir}_pow_no_sil $mfccdir/${datadir}_pow_no_sil || exit 1;
done
fi

if [ $mono_train_sw == 1 ]; then

echo ============================================================================
echo "                   MonoPhone Training                	        "
echo ============================================================================

steps/train_mono.sh  --nj "$train_nj" --cmd "$train_cmd" $train_dir${vad_suffix} $train_lang_dir exp_${feat}/mono || exit 1; 

fi

if [ $mono_test_sw == 1 ]; then

echo ============================================================================
echo "                   MonoPhone Testing             	        "
echo ============================================================================

for datadir in $test_data1${vad_suffix} $test_data2${vad_suffix}; do #

utils/mkgraph.sh --mono $test_lang_dir exp_${feat}/mono exp_${feat}/mono/graph_${datadir} || exit 1;

steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" exp_${feat}/mono/graph_${datadir} \
 data_${feat}/${datadir} exp_${feat}/mono/decode_${datadir} || exit 

done

fi

if [ $tri1_train_sw == 1 ]; then

echo ============================================================================
echo "           tri1 : Deltas + Delta-Deltas Training      "
echo ============================================================================

steps/align_si.sh --boost-silence 1.25 --nj "$train_nj" --cmd "$train_cmd" $train_dir${vad_suffix} $train_lang_dir exp_${feat}/mono exp_${feat}/mono_ali || exit 1; 

for sen in 2000; do 
for gauss in 16; do 
gauss=$(($sen * $gauss)) 

steps/train_deltas.sh --cmd "$train_cmd" $sen $gauss $train_dir${vad_suffix} $train_lang_dir exp_${feat}/mono_ali exp_${feat}/tri1 || exit 1; 

done;done

fi

if [ $tri1_test_sw == 1 ]; then

echo ============================================================================
echo "           tri1 : Deltas + Delta-Deltas  Decoding            "
echo ============================================================================

for datadir in $test_data1${vad_suffix} $test_data2${vad_suffix}; do

for sen in 2000; do
utils/mkgraph.sh $test_lang_dir exp_${feat}/tri1 exp_${feat}/tri1/graph_${datadir} || exit 1;

steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd"  exp_${feat}/tri1/graph_${datadir} \
 data_${feat}/${datadir} exp_${feat}/tri1/decode_${datadir} || exit 1;

#steps/score_kaldi.sh --cmd "run.pl" data_${feat}/${datadir} exp_${feat}/tri1/graph_${datadir} \
# exp_${feat}/tri1/decode_${datadir} || exit 1;

done;done

fi


if [ $tri2_train_sw == 1 ]; then

echo ============================================================================
echo "                 tri2 : LDA + MLLT Training                    "
echo ============================================================================

for sen in 2000; do
steps/align_si.sh --nj "$train_nj" --cmd "$train_cmd" $train_dir${vad_suffix} $train_lang_dir \
 exp_${feat}/tri1 exp_${feat}/tri1_ali || exit 1;

steps/train_lda_mllt.sh --cmd "$train_cmd" --splice-opts "--left-context=3 \
 --right-context=3" $numLeavesMLLT $numGaussMLLT $train_dir${vad_suffix} \
 $train_lang_dir exp_${feat}/tri1_ali exp_${feat}/tri2 || exit 1;

done

fi

if [ $tri2_test_sw == 1 ]; then

echo ============================================================================
echo "                 tri2 : LDA + MLLT Decoding                "
echo ============================================================================

for datadir in $test_data1${vad_suffix} $test_data2${vad_suffix}; do

for sen in 2000; do

utils/mkgraph.sh $test_lang_dir exp_${feat}/tri2 exp_${feat}/tri2/graph_${datadir} || exit 1;

steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" exp_${feat}/tri2/graph_${datadir} \
 data_${feat}/${datadir} exp_${feat}/tri2/decode_${datadir} || exit 1;

#steps/score_kaldi.sh --cmd "run.pl" data_${feat}/${datadir} exp_${feat}/tri2/graph_${datadir} \
# exp_${feat}/tri2/decode_${datadir} || exit 1;

done;done

fi

if [ $tri3_train_sw == 1 ]; then

echo ============================================================================
echo "              tri3 : LDA + MLLT + SAT Training               "
echo ============================================================================


steps/align_si.sh --nj "$train_nj" --cmd "$train_cmd" \
 --use-graphs true $train_dir${vad_suffix} $train_lang_dir exp_${feat}/tri2 exp_${feat}/tri2_ali || exit 1;


# From tri2 system, train tri3 which is LDA + MLLT + SAT.
steps/train_sat.sh --cmd "$train_cmd" \
 $numLeavesSAT $numGaussSAT $train_dir${vad_suffix} $train_lang_dir exp_${feat}/tri2_ali exp_${feat}/tri3 || exit 1;


steps/align_fmllr.sh --nj "$train_nj" --cmd "$train_cmd" \
 $train_dir${vad_suffix} $train_lang_dir exp_${feat}/tri3 exp_${feat}/tri3_ali || exit 1;


fi

if [ $tri3_test_sw == 1 ]; then

echo ============================================================================
echo "              tri3 : LDA + MLLT + SAT Decoding    Start             "
echo ============================================================================

for datadir in $test_data1${vad_suffix} $test_data2${vad_suffix}; do


utils/mkgraph.sh $test_lang_dir exp_${feat}/tri3 exp_${feat}/tri3/graph_${datadir} || exit 1;

steps/decode_fmllr.sh --nj "$decode_nj" --cmd "$decode_cmd" exp_${feat}/tri3/graph_${datadir} \
 data_${feat}/${datadir} exp_${feat}/tri3/decode_${datadir} || exit 1;

#steps/score_kaldi.sh --cmd "run.pl" data_${feat}/${datadir} exp_${feat}/tri3/graph_${datadir} \
# exp_${feat}/tri3/decode_${datadir}

done

fi

if [ $tdnn_train_sw == 1 ]; then

echo ============================================================================
echo "                    	Chain2 TDNN Training                		 "
echo ============================================================================
# DNN hybrid system training parameters
tdnn_stage=22
tdnn_train_iter=45 #default=-10

local/chain2/train_tdnn_1a_vad.sh --stage $tdnn_stage --train_stage $tdnn_train_iter \
 --train_set $train_data${vad_suffix} --gmm tri3 --feat ${feat} || exit 1;
     
fi

if [ $tdnn_test_sw == 1 ]; then

echo ============================================================================
echo "                    	Chain2 TDNN Testing			          "
echo ============================================================================

test_stage=1
mfccdir=${feat}_feats

if [ $test_stage -le 0 ]; then
  
for datadir in $test_data $test_data2; do #test_other
  
    utils/fix_data_dir.sh data_${feat}/$datadir
    utils/copy_data_dir.sh data_${feat}/$datadir data_${feat}/${datadir}_hires || exit 1;
done

for datadir in ${test_data}_hires ${test_data2}_hires; do # $train_data $test_data
	
	utils/fix_data_dir.sh data_${feat}/${datadir}
	utils/validate_data_dir.sh data_${feat}/${datadir}
	utils/copy_data_dir.sh data_${feat}/$datadir data_${feat}/${datadir}_pow || exit 1;
	utils/copy_data_dir.sh data_${feat}/$datadir data_${feat}/${datadir}_ener || exit 1;

	steps/make_mfcc.sh --mfcc-config conf/mfcc_hires_pow.conf --cmd "$train_cmd" --nj 10 \
	 data_${feat}/${datadir}_pow exp_${feat}/make_mfcc/${datadir}_pow \
	 $mfccdir/${datadir}_pow || exit 1;
	 
	steps/make_mfcc.sh --mfcc-config conf/mfcc_hires_ener.conf --cmd "$train_cmd" --nj 10 \
	 data_${feat}/${datadir}_ener exp_${feat}/make_mfcc/${datadir}_ener \
	 $mfccdir/${datadir}_ener || exit 1;
	  
	sid/compute_vad_decision.sh --nj 5 --cmd "$train_cmd" \
    	 data_${feat}/${datadir}_ener exp_${feat}/make_mfcc/${datadir}_ener $mfccdir/${datadir}_ener || exit 1;
    	 
        cp data_${feat}/${datadir}_ener/vad.scp data_${feat}/${datadir}_pow || exit 1;
        utils/fix_data_dir.sh data_${feat}/${datadir}_pow || exit 1;
        
  	local/nnet3/xvector/prepare_feats_for_egs.sh --nj 5 --cmd "$train_cmd" \
    	  data_${feat}/${datadir}_pow data_${feat}/${datadir}_pow_no_sil $mfccdir/${datadir}_pow_no_sil || exit 1;
    	  
    	cp data_${feat}/${datadir}/text data_${feat}/${datadir}_pow_no_sil || exit 1;
    	rm -r data_${feat}/${datadir}
  	utils/copy_data_dir.sh data_${feat}/${datadir}_pow_no_sil data_${feat}/${datadir} || exit 1;

 	steps/compute_cmvn_stats.sh data_${feat}/${datadir} \
 	 exp_${feat}/make_mfcc/${datadir} $mfccdir/${datadir} || exit 1;
done
fi
if [ $test_stage -le 1 ]; then

    for datadir in $test_data $test_data2; do #test_other
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 5 data_${feat}/${datadir}_hires \
     exp_${feat}/nnet3/extractor exp_${feat}/nnet3/ivectors_${datadir}_hires || exit 1;
     
    done
fi

if [ $test_stage -le 2 ]; then
  for datadir in $test_data $test_data2; do
	  utils/mkgraph.sh --self-loop-scale 1.0 --remove-oov \
	   $test_lang_dir exp_${feat}/chain2/tree_sp exp_${feat}/chain2/tree_sp/graph_${datadir} || exit 1;
  done
    
fi

if [ $test_stage -le 3 ];then
  for datadir in $test_data $test_data2; do
	  steps/nnet3/decode.sh \
	  	  --use-gpu true \
		  --acwt 1.0 --post-decode-acwt 10.0 \
		  --extra-left-context 22 \
		  --extra-right-context 16 \
		  --extra-left-context-initial 0 \
		  --extra-right-context-final 0 \
		  --frames-per-chunk 140 \
		  --nj 4 --cmd "$decode_cmd"  --num-threads 4 \
		  --online-ivector-dir exp_${feat}/nnet3/ivectors_${datadir}_hires \
		  exp_${feat}/chain2/tree_sp/graph_${datadir} \
		  data_${feat}/${datadir}_hires \
		  exp_${feat}/chain2/tdnn1a_sp/decode_${datadir} || exit 1;
  done
fi

if [ $test_stage -eq 4 ];then
  for datadir in $test_data $test_data2; do
	  steps/score_kaldi.sh --cmd "run.pl" data_${feat}/${datadir} exp_${feat}/chain2/tree_sp/graph_${datadir} \
	   exp_${feat}/chain2/tdnn1a_sp/decode_${datadir} || exit 1;
  done

fi

fi


if [ $dnn_train_sw == 1 ]; then

echo ============================================================================
echo "                    DNN Training                  "
echo ============================================================================

# DNN hybrid system training parameters
dnn_mem_reqs="mem_free=1.0G,ram_free=1.0G"
dnn_extra_opts="--num_epochs 20 --num-epochs-extra 10 --add-layers-period 1 --shrink-interval 3"

steps/nnet2/train_tanh.sh --stage -10 --mix-up 5000 --initial-learning-rate 0.005 \
  --final-learning-rate 0.0005 --num-hidden-layers 5 --hidden_layer_dim 1024  \
  --num-jobs-nnet "$train_nj" --cmd "$train_cmd" "${dnn_train_extra_opts[@]}" \
  ${train_dir}${vad_suffix} $train_lang_dir exp_${feat}/tri3_ali exp_${feat}/DNN

fi

if [ $dnn_test_sw == 1 ]; then

echo ============================================================================
echo "                    DNN Testing                  "
echo ============================================================================

dnn_mem_reqs="mem_free=1.0G,ram_free=0.2G"
dnn_extra_opts="--num_epochs 20 --num-epochs-extra 10 --add-layers-period 1 --shrink-interval 3"

decode_extra_opts=(--num-threads 6 --parallel-opts "-pe smp 6 -l mem_free=4G,ram_free=0.7G")
for iter in 52 53 54 55 56; do #final 
for datadir in $test_data1${vad_suffix} $test_data2${vad_suffix}; do
steps/nnet2/decode.sh --cmd "$decode_cmd" --nj "$decode_nj" "${decode_extra_opts[@]}" --iter $iter \
 --transform-dir exp_${feat}/tri3/decode_${datadir} exp_${feat}/tri3/graph_${datadir} data_${feat}/${datadir} \
  exp_${feat}/DNN/decode_${datadir}.$iter | tee exp_${feat}/DNN/decode_${datadir}.$iter/decode.log
  
#steps/score_kaldi.sh --cmd "run.pl" $test_dir exp_${feat}/tri3/graph_${datadir} exp_${feat}/DNN/decode_${datadir}
done; done
fi

echo ============================================================================
echo "                     Training Testing Finished                      "
echo ============================================================================
