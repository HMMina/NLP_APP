import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer, StopWordsRemover, HashingTF, IDF, Word2Vec}
import org.apache.spark.ml.Pipeline
import java.io.{File, PrintWriter}
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

object TextPipelineExample {
  def main(args: Array[String]): Unit = {
    val startTime = LocalDateTime.now()
    val formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")
    
    // Tạo thư mục log và results nếu chưa có
    new File("log").mkdirs()
    new File("results").mkdirs()
    
    // Khởi tạo log writer
    val logWriter = new PrintWriter(new File(s"log/pipeline_${startTime.format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))}.log"))
    
    try {
      logWriter.println(s"Pipeline started at: ${startTime.format(formatter)}")
      logWriter.flush()
    // 1. Tạo SparkSession với config tối ưu cho Windows
    val spark = SparkSession.builder()
      .appName("TF-IDF Pipeline Example")
      .master("local[*]") // chạy local
      .config("spark.sql.adaptive.enabled", "true")
      .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
      .config("spark.storage.level", "MEMORY_ONLY")
      .config("spark.local.dir", System.getProperty("java.io.tmpdir"))
      .getOrCreate()

      // Giảm log level để ít error message hơn
      spark.sparkContext.setLogLevel("WARN")

      import spark.implicits._

      logWriter.println("SparkSession created successfully")
      logWriter.flush()

      // 2. Load dữ liệu JSON
      val dataPath = "../../c4-train.00000-of-01024-30K.json"
      logWriter.println(s"Loading data from: $dataPath")
      logWriter.flush()
      
      val df = spark.read.json(dataPath)
        .limit(1000) // chỉ lấy 1000 dòng cho nhanh

      // Giả sử cột văn bản tên là "text"
      val textDF = df.select("text").na.drop()
      
      val recordCount = textDF.count()
      logWriter.println(s"Loaded $recordCount records")
      logWriter.flush()
      
      // 3. Tokenization (RegexTokenizer)
      val tokenizer = new RegexTokenizer()
        .setInputCol("text")
        .setOutputCol("tokens")
        .setPattern("\\s+|[.,;!?()\"']") // tách theo ký tự không phải chữ

      // EXERCISE 1: Uncomment line below and comment RegexTokenizer above
      // val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("tokens")

      logWriter.println("Tokenizer configured")
      logWriter.flush()

      // 4. StopWordsRemover
      val remover = new StopWordsRemover()
        .setInputCol("tokens")
        .setOutputCol("filtered")

      logWriter.println("StopWordsRemover configured")
      logWriter.flush()

      // 5. HashingTF (Term Frequency) - sử dụng 20000 features như yêu cầu
      // EXERCISE 2: Change numFeatures from 20000 to 1000
      val hashingTF = new HashingTF()
        .setInputCol("filtered")
        .setOutputCol("rawFeatures")
        .setNumFeatures(20000) // số chiều vector

      logWriter.println("HashingTF configured with 20000 features")
      logWriter.flush()

      // 6. IDF (Inverse Document Frequency)
      val idf = new IDF()
        .setInputCol("rawFeatures")
        .setOutputCol("features")

      logWriter.println("IDF configured")
      logWriter.flush()

      // EXERCISE 4: Alternative Word2Vec implementation (comment out HashingTF + IDF above)
      /*
      val word2Vec = new Word2Vec()
        .setInputCol("filtered")
        .setOutputCol("features")
        .setVectorSize(100)     // Word2Vec vector size
        .setMinCount(0)         // Minimum word frequency
        
      logWriter.println("Word2Vec configured")
      logWriter.flush()
      */

      // 7. Xây dựng Pipeline
      val pipeline = new Pipeline()
        .setStages(Array(tokenizer, remover, hashingTF, idf))
        
      // EXERCISE 4: For Word2Vec, use this pipeline instead:
      // val pipeline = new Pipeline()
      //   .setStages(Array(tokenizer, remover, word2Vec))

      logWriter.println("Pipeline configured")
      logWriter.flush()

      // 8. Fit pipeline và transform dữ liệu
      logWriter.println("Starting pipeline fitting...")
      logWriter.flush()
      
      val model = pipeline.fit(textDF)
      val result = model.transform(textDF)

      logWriter.println("Pipeline fitting completed")
      logWriter.flush()

      // 9. Hiển thị kết quả
      println("=== TF-IDF Pipeline Results ===")
      result.select("text", "tokens", "filtered", "features").show(5, truncate = false)
      
      // 10. Lưu kết quả vào file theo yêu cầu
      logWriter.println("Saving results to file...")
      logWriter.flush()
      
      // Lưu kết quả vào file với tên chính xác như yêu cầu
      val outputFileName = "results/lab17_pipeline_output.txt"
      
      // Tạo file writer cho kết quả
      val resultWriter = new PrintWriter(new File(outputFileName))
      
      try {
        // Thu thập kết quả và ghi vào file
        val resultData = result.select("text", "features").collect()
        
        resultWriter.println("=== TF-IDF Pipeline Output ===")
        resultWriter.println(s"Pipeline executed at: ${startTime.format(formatter)}")
        resultWriter.println(s"Total documents processed: ${resultData.length}")
        resultWriter.println(s"Feature vector size: ${hashingTF.getNumFeatures}")
        resultWriter.println("=" * 50)
        
        resultData.zipWithIndex.foreach { case (row, index) =>
          resultWriter.println(s"Document ${index + 1}:")
          resultWriter.println(s"Text: ${row.getAs[String]("text").take(100)}...")
          resultWriter.println(s"Features: ${row.getAs[org.apache.spark.ml.linalg.Vector]("features")}")
          resultWriter.println("-" * 30)
        }
        
        resultWriter.println("=== End of Output ===")
        
      } finally {
        resultWriter.close()
      }

      logWriter.println(s"Results saved successfully to $outputFileName")
      logWriter.flush()
      
      // 11. Một số thống kê
      val finalCount = result.count()
      println(s"Total documents processed: $finalCount")
      println(s"Vocabulary size (HashingTF): ${hashingTF.getNumFeatures}")
      
      logWriter.println(s"Total documents processed: $finalCount")
      logWriter.println(s"Vocabulary size (HashingTF): ${hashingTF.getNumFeatures}")
      
      val endTime = LocalDateTime.now()
      logWriter.println(s"Pipeline completed successfully at: ${endTime.format(formatter)}")
      logWriter.flush()
      
      // 12. Đóng Spark session một cách an toàn
      try {
        spark.stop()
        logWriter.println("Spark session closed successfully")
      } catch {
        case e: Exception => 
          println(s"Warning during Spark shutdown: ${e.getMessage}")
          logWriter.println(s"Warning during Spark shutdown: ${e.getMessage}")
      }
      
    } catch {
      case e: Exception =>
        logWriter.println(s"Error occurred: ${e.getMessage}")
        logWriter.println(s"Stack trace: ${e.getStackTrace.mkString("\n")}")
        throw e
    } finally {
      logWriter.close()
    }
  }
}
