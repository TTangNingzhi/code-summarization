# Code Summarization

Course project for CSE 60657 Natural Language Processing, Fall 2023, University of Notre Dame.

## Dataset

The dataset is downloaded from [NeuralCodeSum](https://github.com/wasiahmad/NeuralCodeSum).

## Model

Vector Space Model (VSM)

- VSM-RAND: rand.py
- VSM-TFIDF: tfidf.py

## Prompt

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

[XXXXX]

Summaries (You need to write):
```

## Contact

Ningzhi Tang (ntang@nd.edu) and Gelei Xu (gxu4@nd.edu)

## Acknowledgement

Thanks to ChatGPT and PyTorch
tutorial [Language Translation with nn.Transformer and TorchText](https://pytorch.org/tutorials/beginner/translation_transformer.html)
for the code reference of implementing Transformer.