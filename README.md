# Code Summarization

Course project for CSE 60657 Natural Language Processing, Fall 2023, University of Notre Dame.

## Dataset

The dataset is downloaded from [NeuralCodeSum](https://github.com/wasiahmad/NeuralCodeSum). You could set the dataset by
modifying the `language` variable in `config.py` as `java` or `python`.

## Model

- VSM-RAND (Vector Space Model): rand.py
- VSM-TFIDF: tfidf.py
- Transformer: transformer.py
- Transformer with TFIDF: transformer_tfidf.py
- GPT-4: gpt.py

You could directly run the corresponding python file to train the model.

## Data Preprocessing

Since a small portion of the code/comments are too long and seem to be wired (e.g., the comment is just a copy of the
code), the cuda memory is not enough to train such pieces of data, i.e., large input embeddings. Thus, we filter out
those data that are too long. The threshold is set as 1600 tokens, which larger than 99.9% quantile of the distribution
of code snippets length. Thus, less than 0.1% of the data are filtered out but the training process can be finished.

## Prompt

Below is the prompt for the GPT-4 model (Java dataset). For convenience, we just generate the first 100 lines of the
test dataset for evaluation.

```
You are a software engineer working code summarization. Given a set of code snippets, you need to write a short summary for each of them in natural language. For example,

Code snippets:

@ Override public int run Command ( boolean merge Error Into Output , String ... commands ) throws IO Exception , Interrupted Exception { return run Command ( merge Error Into Output , new Array List < String > ( Arrays . as List ( commands ) ) ) ; }
private int find PLV ( int M Price List ID ) { Timestamp price Date = null ; String date Str = Env . get Context ( Env . get Ctx ( ) , p Window No , STRING ) ; if ( date Str != null && date Str . length ( ) > NUM ) price Date = Env . get Context As Date ( Env . get Ctx ( ) , p Window No , STRING ) ; else { date Str = Env . get Context ( Env . get Ctx ( ) , p Window No , STRING ) ; if ( date Str != null && date Str . length ( ) > NUM ) price Date = Env . get Context As Date ( Env . get Ctx ( ) , p Window No , STRING ) ; } if ( price Date == null ) price Date = new Timestamp ( System . current Time Millis ( ) ) ; log . config ( STRING + M Price List ID + STRING + price Date ) ; int ret Value = NUM ; String sql = STRING + STRING + STRING + STRING + STRING + STRING ; try { Prepared Statement pstmt = DB . prepare Statement ( sql , null ) ; pstmt . set Int ( NUM , M Price List ID ) ; Result Set rs = pstmt . execute Query ( ) ; while ( rs . next ( ) && ret Value == NUM ) { Timestamp pl Date = rs . get Timestamp ( NUM ) ; if ( ! price Date . before ( pl Date ) ) ret Value = rs . get Int ( NUM ) ; } rs . close ( ) ; pstmt . close ( ) ; } catch ( SQL Exception e ) { log . log ( Level . SEVERE , sql , e ) ; } Env . set Context ( Env . get Ctx ( ) , p Window No , STRING , ret Value ) ; return ret Value ; }
public static boolean memory Is Low ( ) { return available Memory ( ) * NUM < RUNTIME . total Memory ( ) * NUM ; }
public String describe Attributes ( ) { String Builder sb = new String Builder ( ) ; sb . append ( STRING ) ; boolean first = BOOL ; for ( Object key : attributes . key Set ( ) ) { if ( first ) { first = BOOL ; } else { sb . append ( STRING ) ; } sb . append ( key ) ; sb . append ( STRING ) ; sb . append ( attributes . get ( key ) ) ; } sb . append ( STRING ) ; return sb . to String ( ) ; }
public static byte [ ] next Bytes ( byte [ ] buffer ) { s Random . next Bytes ( buffer ) ; return buffer ; }

Summaries:

runs a command on the command line synchronously .
find price list version and update context
returns true if less then 5 % of the available memory is free .
returns a string representation of the object ' s current attributes
fill the given buffer with random bytes .

Now, it is your turn. Please follow the example above, just write a short summary with all words in lower case. The punctuation marks should be separated from the words. For example, `returns true if less then 5 % of the available memory is free .` is a good summary. Do not add any extra things.

Code snippets:

[INPUT CODE SNIPPETS HERE]

Summaries (You need to write):
```

## Acknowledgement

Thanks to ChatGPT and PyTorch
tutorial [Language Translation with nn.Transformer and TorchText](https://pytorch.org/tutorials/beginner/translation_transformer.html)
for the code reference of implementing Transformer.

## Contact

Ningzhi Tang (ntang@nd.edu) and Gelei Xu (gxu4@nd.edu)

