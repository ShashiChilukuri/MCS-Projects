package cse512

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions._

object HotcellAnalysis {
  Logger.getLogger("org.spark_project").setLevel(Level.WARN)
  Logger.getLogger("org.apache").setLevel(Level.WARN)
  Logger.getLogger("akka").setLevel(Level.WARN)
  Logger.getLogger("com").setLevel(Level.WARN)

  def runHotcellAnalysis(spark: SparkSession, pointPath: String): DataFrame =
  {
    // Load the original data FROM a data source
    var pickupInfo = spark.read.format("com.databricks.spark.csv").option("delimiter",";").option("header","false").load(pointPath);
    pickupInfo.createOrReplaceTempView("nyctaxitrips")
    //pickupInfo.show()

    // ASsign cell coordinates bASed on pickup points
    spark.udf.register("CalculateX",(pickupPoint: String)=>((
      HotcellUtils.CalculateCoordinate(pickupPoint, 0)
      )))
    spark.udf.register("CalculateY",(pickupPoint: String)=>((
      HotcellUtils.CalculateCoordinate(pickupPoint, 1)
      )))
    spark.udf.register("CalculateZ",(pickupTime: String)=>((
      HotcellUtils.CalculateCoordinate(pickupTime, 2)
      )))
    pickupInfo = spark.sql("SELECT CalculateX(nyctaxitrips._c5),CalculateY(nyctaxitrips._c5), CalculateZ(nyctaxitrips._c1) FROM nyctaxitrips")
    var newCoordinateName = Seq("x", "y", "z")
    pickupInfo = pickupInfo.toDF(newCoordinateName:_*)
    //pickupInfo.show()

    // Define the min and max of x, y, z
    val minX = -74.50/HotcellUtils.coordinateStep
    val maxX = -73.70/HotcellUtils.coordinateStep
    val minY = 40.50/HotcellUtils.coordinateStep
    val maxY = 40.90/HotcellUtils.coordinateStep
    val minZ = 1
    val maxZ = 31
    val numCells = (maxX - minX + 1)*(maxY - minY + 1)*(maxZ - minZ + 1)


    pickupInfo.createOrReplaceTempView("nyctaxitripsNewView")

    //function to check if the point is inside the cube boundary.
    spark.udf.register("IsCellInBounds", (x: Double, y:Double, z:Int) =>
      (x >= minX) && (x <= maxX) && (y >= minY) && (y <= maxY) && (z >= minZ) && (z <= maxZ))
    //Square function
    spark.udf.register("squared", (inputX: Int) => HotcellUtils.squared(inputX))


    // Use the function just created to check if the point is inside the boundary.
    val pointInsideCell = spark.sql("SELECT x,y,z FROM nyctaxitripsNewView " +
      "WHERE IsCellInBounds(x, y, z) "
    ).persist()
    pointInsideCell.createOrReplaceTempView("pointInsideCell")

    //pickUpLocationAndSum
    // x,y,z => Cell, numPoint => total of pickup points
    val pickUpLocationAndSum = spark.sql("SELECT x,y,z,  COUNT(*) AS numPoints " +
      "FROM pointInsideCell " +
      "GROUP BY x,y,z "
    ).persist()
    pickUpLocationAndSum.createOrReplaceTempView("pickUpLocationAndSum")


    val allPoints = spark.sql("SELECT " +
      "COUNT(*), SUM(numPoints) , SUM(squared(numPoints)) " +
      "FROM pickUpLocationAndSum")
    allPoints.createOrReplaceTempView("allPoints")


    val sigmaXj = allPoints.first().getLong(1)// sigma(Xj)
  val sigmaSquareXj2 = allPoints.first().getDouble(2) // sigma(Xj^2)


    // X-bar and S
    //use sigma(Xj), sigma(Xj^2), X-bar
    val Xbar = HotcellUtils.X(sigmaXj, numCells)
    val SD = HotcellUtils.BigS(sigmaSquareXj2,numCells,Xbar)


    spark.udf.register("CountNeighbours", (minX: Int, minY: Int, minZ: Int, maxX: Int, maxY: Int, maxZ: Int, totalX: Int, totalY: Int, totalZ: Int)
    => ((HotcellUtils.CountNeighbours(minX, minY, minZ, maxX, maxY, maxZ, totalX, totalY, totalZ))))

    val Neighbours = spark.sql("SELECT " +
      "CountNeighbours("+minX + "," + minY + "," + minZ + "," + maxX + "," + maxY + "," + maxZ + ","
      + "table1.x,table1.y,table1.z) AS totalNeighbours, " +   // function to get the number of neighbours of x,y,z
      "COUNT(*) AS neighboursWithValidPoints, " +  // COUNT the neighbours with atLeaSt one pickup point
      "table1.x AS x, " + "table1.y AS y, " + "table1.z AS z, " +
      "SUM(table2.numPoints) AS sumAllNeighboursPoints " +
      "FROM pickUpLocationAndSum AS table1,  pickUpLocationAndSum AS table2 " +    // two tables to join
      "WHERE (table2.x = table1.x or    table2.x = table1.x+1 or table2.x = table1.x-1) " +
      "AND   (table2.y = table1.y or    table2.y = table1.y+1 or table2.y = table1.y-1) " +
      "AND   (table2.z = table1.z or    table2.z = table1.z+1 or table2.z = table1.z-1) " +   //join condition
      "GROUP BY table1.x, table1.y, table1.z "
    ).persist()

    Neighbours.createOrReplaceTempView("NView")

    spark.udf.register("GIScore", ( numcells: Int, X:Double, sd: Double, totalNeighbours: Int, sumAllNeighboursPoints: Int) =>
      HotcellUtils.GIScore( numcells, X, sd, totalNeighbours, sumAllNeighboursPoints))

    val GIStat = spark.sql("SELECT x, y, z, " +
      "GIScore(" +numCells+ ", " + Xbar + ", " + SD + ", totalNeighbours, sumAllNeighboursPoints) AS score " +
      "FROM NView " +
      "ORDER BY score desc")
    GIStat.createOrReplaceTempView("finalView")
    // GIStat.show()

    val finalResult = spark.sql("SELECT x,y,z FROM finalView ORDER BY score desc, x desc, y asc, z desc")

    return finalResult
  }
}