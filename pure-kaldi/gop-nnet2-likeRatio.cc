#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "hmm/transition-model.h"
#include "nnet2/am-nnet.h"
#include "nnet2/nnet-compute.h"

//define a new GOP class for writting according to kaldi IO fasion
typedef std::vector<std::pair<int32, double>> GOPres;
class GOPresHolder {
  public:
    typedef GOPres T; //GOPresHolder type T will be referenced by the TableWriter template class 
    GOPresHolder() { t_ = NULL; }
    static bool Write(std::ostream &os, bool binary, const T &t) {
      if (binary) {
        KALDI_ERR << "GOP writer doesn't support binaray write, use text in command line";
      } else {
        // Text-mode output. 
        os << '\n';
        for (auto it = t.begin(); it != t.end(); ++it){
          os << std::distance(t.begin(), it) <<  "\t" << it->first << "\t" << it->second << "\n";
        }
        if (os.fail())
          KALDI_WARN << "Stream failure detected.";
        // Write another newline as a terminating character.  The read routine will
        // detect this [this is a Kaldi mechanism, not somethig in the original
        // OpenFst code].
        os << '\n';
        return os.good();
      }
    }
    
    bool Read(std::istream &is) {
          Clear(); // in case anything currently stored.
          int c = is.peek();
          if (c == -1) {
            KALDI_WARN << "End of stream detected";
            return false;
          } else if (isspace(c)) { // The text form of the lattice begins
            // with space (normally, '\n'), so this means it's text (the binary form
            // cannot begin with space because it starts with the FST Type() which is not
            // space).
            return ReadGop(is, false, &t_);
          } else if (c != 214) { // 214 is first char of FST magic number,
            // on little-endian machines which is all we support (\326 octal)
            KALDI_WARN << "Reading compact lattice: does not appear to be an FST "
                      << " [non-space but no magic number detected], file pos is "
                      << is.tellg();
            return false;
          } else {
            return ReadGop(is, true, &t_);
          }
     }

    bool ReadGop(std::istream &is, bool binary, GOPres **gopRes) {
      KALDI_ASSERT(*gopRes == NULL);
      if (binary) {
        KALDI_ERR << "GOP reader doesn't support binaray read currently, use text in command line";
      }
      else{
        // The next line would normally consume the \r on Windows, plus any
        // extra spaces that might have got in there somehow.
        while (std::isspace(is.peek()) && is.peek() != '\n') is.get(); 
        if (is.peek() == '\n') is.get(); // consume the newline.
        else { // saw spaces but no newline.. this is not expected.
          KALDI_WARN << "Reading GOP text: unexpected sequence of spaces "
                 << " at file position " << is.tellg();
          return false;
        } 

        *gopRes = new GOPres;  // don't forget to free after using it
        size_t nline = 0;
        std::string line;
        std::vector<string> col;
        string separator = FLAGS_fst_field_separator + "\r\n";
        kaldi::SplitStringToVector(line, separator.c_str(), true, &col);
        while (std::getline(is, line)) {
          nline++;
          if (col.size() == 0) break; // Empty line is a signal to stop, in our
          // archive format.
          if (col.size() != 2) {
            KALDI_WARN << "Reading GOP: bad line in GOP: " << line;
            delete *gopRes;
            *gopRes = NULL;
            break;
          }
          int32_t phoneme;
          if (!kaldi::ConvertStringToInteger(col[0], &phoneme)){
              KALDI_WARN << "Reading GOP: bad line in GOP: " << line;
              delete *gopRes;
              *gopRes = NULL;
              break;
          }
          double_t gScore;
          if (!kaldi::ConvertStringToReal(col[1], &gScore)){
              KALDI_WARN << "Reading GOP: bad line in GOP: " << line;
              delete *gopRes;
              *gopRes = NULL;
              break;
          }
          (*gopRes)->push_back(std::make_pair(phoneme, gScore));
        }
      }
      return (*gopRes != NULL);
    }

    void Clear() {  delete t_; t_ = NULL; }
    
  private:
    T *t_;
};

typedef kaldi::TableWriter<GOPresHolder> GOPWriter;
typedef kaldi::SequentialTableReader<GOPresHolder> GOPReader;

int main(int argc, char *argv[]) {

    try{
        using namespace kaldi;
        using namespace kaldi::nnet2;

    const char *usage =
        "calculating the GOP based on the given alignment(raw alignment with transition ids) and neural network posteriors (frame-wise likelihood ratio)\n"
        "Usgae: gop-nnet2-likeRatio NNModel-in feats-rspecifier ali-rspecifier gop-wspecifier";

    ParseOptions po(usage);
    //BaseFloat lm2acoustic_scale = 0.0;
    //po.Register("lm2acoustic-scale", &lm2acoustic_scale, "Add this times original LM costs to acoustic costs");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }
    #if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId("optional");
    #endif

    std::string nnet_rxfilename = po.GetArg(1),
        feats_rspecifier = po.GetArg(2), 
        ali_rspecifier = po.GetArg(3), //read the tran-id alignment file for grouping the gop scores and getting the canonical pdf-id sequence 
        gop_wspecifier= po.GetArg(4);

  
    TransitionModel trans_model;
    AmNnet am_nnet;
    {
      bool binary_read;
      Input ki(nnet_rxfilename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_nnet.Read(ki.Stream(), binary_read);
    }

    SequentialBaseFloatCuMatrixReader feature_reader(feats_rspecifier);
    RandomAccessInt32VectorReader ali_reader(ali_rspecifier);
    GOPWriter gop_writer(gop_wspecifier);



    for (; !feature_reader.Done(); feature_reader.Next()) {
      string uttid = feature_reader.Key();
      if (!ali_reader.HasKey(uttid)){
        KALDI_WARN << "the utt " << uttid << " can not be found in the alignment, skipped";
        continue;   
      }
      KALDI_LOG << "processing " << uttid;

      //Step 1: get the tran-id sequence and segmentation vector from alignment
      const std::vector<int32> &vec_time = ali_reader.Value(uttid);
      int length = vec_time.size();
      std::vector<int32> vec_pdf(length);
      std::vector<std::pair<int32, int32>> vec_seg; //pair<phoneid, start_pos>
      int last_phone = -1;
      for (int i = 0; i < length; ++i){
        int cur_phone = trans_model.TransitionIdToPhone(vec_time[i]);
        if(cur_phone != last_phone){
          vec_seg.push_back(std::make_pair(cur_phone, i));
          last_phone = cur_phone;
        }
        vec_pdf[i] = trans_model.TransitionIdToPdfArray()[vec_time[i]];
      }

      //Step 2: compute the likelihood matrix
      const CuMatrix<BaseFloat> &feats = feature_reader.Value();
      int32 output_frames = feats.NumRows(), output_dim = am_nnet.GetNnet().OutputDim();
      //We always pad the input for GOP calculation
      //if (!pad_input)
      //  output_frames -= nnet.LeftContext() + nnet.RightContext();

      if ( output_frames != length){
        KALDI_ERR << "decoding length is not the same as alignment length, interrupted: " << output_frames << " vs " <<length;
      }
      if (output_frames <= 0) {
        KALDI_WARN << "Skipping utterance " << uttid << " because output "
                   << "would be empty.";
        continue;
      }

      CuMatrix<BaseFloat> like_mat(output_frames, output_dim);
      //We always pad the input for GOP calculation
      NnetComputation(am_nnet.GetNnet(), feats, true, &like_mat);
      CuVector<float> prior_vec(am_nnet.Priors());
      CuMatrix<float> prior_mat(output_frames, output_dim);
      prior_mat.CopyRowsFromVec(prior_vec);
      like_mat.DivElements(prior_mat);
      like_mat.ApplyLog();
      //like_mat.LogSoftMaxPerRow(like_mat); //renormalize before applying log, to ensure the same sign
      like_mat.Scale(-1); //negate it so we need to compute GOP with (Denom - Numerator) and it will be always less than zero 

      //Step 3: compute the GOP
      //numerator
      CuMatrix<float> numerator_mask(output_frames, output_dim);
      CuMatrix<float> numerator_mat(like_mat);
      for(int i=0; i<output_frames; i++){
        numerator_mask.Row(i).SetZero();
        numerator_mask.Row(i)(vec_pdf[i]) = 1;
      }
      numerator_mat.MulElements(numerator_mask);

      //CuVector<float> one_vec(output_dim);
      //one_vec.Set(1);
      CuVector<float> numerator_vec(output_frames);
      numerator_vec.SetZero();
      //numerator_vec.AddMatVec(1, numerator_mat, kNoTrans, one_vec, 1);
      numerator_vec.AddColSumMat(1, numerator_mat);

      //denominator
      CuVector<float> denom_vec(output_frames);
      for (int i=0; i < output_frames; i++ ){
        denom_vec(i) = like_mat.Row(i).Min();
      }
    
      //GOP     
      GOPres results;
      for(int i = 0, last_pos = 0; i < vec_seg.size(); i++){
        int32 p_id = vec_seg[i].first;
        int32 len_seg = -1;
        double gop_score = 0;
        if(i ==  vec_seg.size() - 1){
        	len_seg = output_frames - vec_seg[i].second;
		      denom_vec.Range(last_pos, len_seg).AddVec(-1, numerator_vec.Range(last_pos, len_seg));
        	gop_score = denom_vec.Range(last_pos, len_seg).Sum() / len_seg;
        }
        else{
          len_seg = vec_seg[i+1].second - last_pos;
          denom_vec.Range(last_pos, len_seg).AddVec(-1, numerator_vec.Range(last_pos, len_seg));
          gop_score = denom_vec.Range(last_pos, len_seg).Sum() / len_seg;
          last_pos = vec_seg[i+1].second;
        }	
	    //KALDI_LOG << " " << ;
        results.push_back(std::make_pair(p_id, gop_score));
      }
      //write out
      gop_writer.Write(uttid, results);
    }

    }catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
