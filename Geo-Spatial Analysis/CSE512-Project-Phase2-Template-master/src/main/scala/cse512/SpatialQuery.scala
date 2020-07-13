package cse512

import org.apache.spark.sql.SparkSession

object SpatialQuery extends App{

  def runRangeQuery(spark: SparkSession, arg1: String, arg2: String): Long = {

    val pointDf = spark.read.format("com.databricks.spark.csv").option("delimiter","\t").option("header","false").load(arg1);
    pointDf.createOrReplaceTempView("point")

    // YOU NEED TO FILL IN THIS USER DEFINED FUNCTION
    spark.udf.register("ST_Contains",(queryRectangle:String, pointString:String)=>(ST_contain(pointString, queryRectangle)))

    val resultDf = spark.sql("select * from point where ST_Contains('"+arg2+"',point._c0)")
    resultDf.show()

    return resultDf.count()
  }

  def runRangeJoinQuery(spark: SparkSession, arg1: String, arg2: String): Long = {

    val pointDf = spark.read.format("com.databricks.spark.csv").option("delimiter","\t").option("header","false").load(arg1);
    pointDf.createOrReplaceTempView("point")

    val rectangleDf = spark.read.format("com.databricks.spark.csv").option("delimiter","\t").option("header","false").load(arg2);
    rectangleDf.createOrReplaceTempView("rectangle")

    // YOU NEED TO FILL IN THIS USER DEFINED FUNCTION
    spark.udf.register("ST_Contains",(queryRectangle:String, pointString:String)=>(ST_contain(pointString, queryRectangle)))

    val resultDf = spark.sql("select * from rectangle,point where ST_Contains(rectangle._c0,point._c0)")
    resultDf.show()

    return resultDf.count()
  }

  def runDistanceQuery(spark: SparkSession, arg1: String, arg2: String, arg3: String): Long = {

    val pointDf = spark.read.format("com.databricks.spark.csv").option("delimiter","\t").option("header","false").load(arg1);
    pointDf.createOrReplaceTempView("point")

    // YOU NEED TO FILL IN THIS USER DEFINED FUNCTION
    spark.udf.register("ST_Within",(pointString1:String, pointString2:String, distance:Double)=>(ST_Within(pointString1, pointString2, distance)))

    val resultDf = spark.sql("select * from point where ST_Within(point._c0,'"+arg2+"',"+arg3+")")
    resultDf.show()

    return resultDf.count()
  }

  def runDistanceJoinQuery(spark: SparkSession, arg1: String, arg2: String, arg3: String): Long = {

    val pointDf = spark.read.format("com.databricks.spark.csv").option("delimiter","\t").option("header","false").load(arg1);
    pointDf.createOrReplaceTempView("point1")

    val pointDf2 = spark.read.format("com.databricks.spark.csv").option("delimiter","\t").option("header","false").load(arg2);
    pointDf2.createOrReplaceTempView("point2")

    // YOU NEED TO FILL IN THIS USER DEFINED FUNCTION
    spark.udf.register("ST_Within",(pointString1:String, pointString2:String, distance:Double)=>(ST_Within(pointString1, pointString2, distance)))
    val resultDf = spark.sql("select * from point1 p1, point2 p2 where ST_Within(p1._c0, p2._c0, "+arg3+")")
    resultDf.show()

    return resultDf.count()
  }

  def ST_contain(pointString:String, queryRectangle:String) : Boolean = {
    //splitting point x, y coordinates into p_x, p_y
    val p = pointString.split(",")
    val p_x = p(0).trim().toDouble
    val p_y = p(1).trim().toDouble

    //splitting rectangle x1, y1, x2, y2 coordinates into r_x_1, r_x_2, r_x_3, r_x_4
    val r = queryRectangle.split(",")
    val r_x_1 = r(0).trim().toDouble
    val r_y_1 = r(1).trim().toDouble
    val r_x_2 = r(2).trim().toDouble
    val r_y_2 = r(3).trim().toDouble

    //Find min and max values between x1 & x2
    var min_x: Double = 0
    var max_x: Double = 0
    if(r_x_1 < r_x_2) {
      min_x = r_x_1
      max_x = r_x_2
    }
    else {
      min_x = r_x_2
      max_x = r_x_1
    }
    //Find min and max values between y1 & y2
    var min_y: Double = 0
    var max_y: Double = 0
    if(r_y_1 < r_y_2) {
      min_y = r_y_1
      max_y = r_y_2
    } else {
      min_y = r_y_2
      max_y = r_y_1
    }
    // Checking if the given point's x-coordinate p_x falls between min_x & max_x and
    // y-coordinate p_y falls between min_y && max_y
    if(p_x >= min_x && p_x <= max_x && p_y >= min_y && p_y <= max_y) {
      return true
    } else {
      return false
    }
  }

  def ST_Within(pointString1: String, pointString2: String, distance:Double): Boolean = {
    //splitting point1 x, y coordinates into p_x_1, p_y_1
    val p1 = pointString1.split(",")
    val p_x_1 = p1(0).trim().toDouble
    val p_y_1 = p1(1).trim().toDouble

    //splitting point2 x, y coordinates into p_x_2, p_y_2
    val p2 = pointString2.split(",")
    val p_x_2 = p2(0).trim().toDouble
    val p_y_2 = p2(1).trim().toDouble

    //calculating euclidean distance
    val euclidean_dist = scala.math.pow(scala.math.pow((p_x_1 - p_x_2), 2) + scala.math.pow((p_y_1 - p_y_2), 2), 0.5)

    //return if euclidean distance is less than given distance
    return euclidean_dist <= distance
  }
}
