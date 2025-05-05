ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.12.18"

crossScalaVersions := Seq("2.12.18", "2.13.8")

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.4.4",
  "org.apache.spark" %% "spark-sql" % "3.4.4",
  "org.apache.spark" %% "spark-mllib" % "3.4.4",
  "org.apache.spark" %% "spark-streaming" % "3.4.4",
  "org.apache.spark" %% "spark-graphx" % "3.4.4"
)

javaOptions += "--add-exports java.base/sun.nio.ch=ALL-UNNAMED"
javaOptions += "-Xmx4G"
javaOptions += "-Duser.timezone=UTC"

fork in run := true

lazy val root = (project in file("."))
  .settings(
    name := "learn-spark-with-scala"
  )

