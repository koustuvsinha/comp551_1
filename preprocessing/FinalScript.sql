--Steps
-- Upload the row transformed CSV into a table (total count 26961)
USE ML
GO

SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO



CREATE TABLE [dbo].Finaldataset4(
	[PARTICIPANT_ID] INT NULL,
	[EVENT_DATE] VARCHAR(100) NULL,
	[EVENT_NAME] [varchar](100) NULL,
	[EVENT_TYPE] [varchar](100) NULL,
	[TIME] [varchar](50) NULL,
	[CATEGORY] [varchar](50) NULL,
	[GENDER] [varchar](50) NULL,
	[EVENT_DT] [varchar](50) NULL,
	[DAY] [varchar](50) NULL,
	[YEAR] [varchar](50) NULL,
	[DAY_OF_WEEK] [varchar](50) NULL,
	[MONTH] [varchar](50) NULL,
	[ID] INT NULL,
	[LOCATION] [varchar](100) NULL,
	[LABEL] INT NULL
) ON [PRIMARY]

GO

sp_configure 'Ad Hoc Distributed Queries',1
GO
sp_configure 'xp_cmdshell',1
GO
reconfigure
GO

-- This query will insert the CSV into the table.
INSERT INTO ML.dbo.Finaldataset4 (LABEL,PARTICIPANT_ID,EVENT_DATE,EVENT_NAME,EVENT_TYPE,TIME,CATEGORY,
GENDER,EVENT_DT,DAY,YEAR,DAY_OF_WEEK,MONTH,LOCATION)
select LABEL,PARTICIPANT_ID,EVENT_DATE,EVENT_NAME,EVENT_TYPE,TIME,CATEGORY,
GENDER,EVENT_DT,DAY,YEAR,DAY_OF_WEEK,MONTH,LOCATION
from openrowset('MSDASQL'
               ,'Driver={Microsoft Access Text Driver (*.txt, *.csv)};'
               ,'select * from G:\comp551_1\FinalData.CSV')


SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

--DROP TABLE [DataSet3]

CREATE TABLE [dbo].[DataSet4](
	[PARTICIPANT_ID] INT NULL,
	[EVENT_DATE] VARCHAR(100) NULL,
	[EVENT_NAME] [varchar](100) NULL,
	[EVENT_TYPE] [varchar](100) NULL,
	[TIME] [varchar](50) NULL,
	[CATEGORY] [varchar](50) NULL,
	[GENDER] [varchar](50) NULL,
	[EVENT_DT] [varchar](50) NULL,
	[DAY] [varchar](50) NULL,
	[YEAR] [varchar](50) NULL,
	[DAY_OF_WEEK] [varchar](50) NULL,
	[MONTH] [varchar](50) NULL,
	[ID] INT NULL,
	[LOCATION] [varchar](100) NULL,
	[LABEL] INT NULL
) ON [PRIMARY]

GO


INSERT INTO DataSet4(LABEL,PARTICIPANT_ID,EVENT_DATE,EVENT_NAME,EVENT_TYPE,TIME,CATEGORY,
GENDER,EVENT_DT,DAY,YEAR,DAY_OF_WEEK,MONTH,LOCATION)
SELECT L.LABEL,L.PARTICIPANT_ID,L.EVENT_DATE,
L.EVENT_NAME,L.EVENT_TYPE,L.TIME,
L.CATEGORY,L.GENDER,L.EVENT_DT,L.DAY,L.YEAR,L.DAY_OF_WEEK,L.MONTH,L.LOCATION 
FROM ML..Finaldataset4 L
WHERE LABEL = '1'

UPDATE L
SET LABEL = 1
FROM ML..DataSet4 L
WHERE L.LABEL = 0
AND L.EVENT_TYPE IN ('Half-Marathon','Demi-Marathon','10K Marathon','5K',
'5K Road Race','30K Marathon','Half-Marathon - Demi-Marathon') 

INSERT INTO DataSet4(LABEL,PARTICIPANT_ID,EVENT_DATE,EVENT_NAME,EVENT_TYPE,TIME,CATEGORY,
GENDER,EVENT_DT,DAY,YEAR,DAY_OF_WEEK,MONTH,LOCATION)
SELECT L.LABEL,L.PARTICIPANT_ID,L.EVENT_DATE,
L.EVENT_NAME,L.EVENT_TYPE,L.TIME,
L.CATEGORY,L.GENDER,L.EVENT_DT,L.DAY,L.YEAR,L.DAY_OF_WEEK,L.MONTH,L.LOCATION 
FROM ML..Finaldataset4 L
WHERE LABEL = '0'
GO
---------------------------------------------------------------------------


-- Add extra columns into the table (features) TimeInSeconds,Distance,Age,Gender

USE ML
GO

SELECT * 
INTO DataSet4Modified
FROM DataSet4
GO

ALTER TABLE DataSet4Modified
ADD TimePerKm DECIMAL(10,0),
	TimeInSeconds INT NULL,
	EveTyp INT NULL,
	Age1 INT NULL,
	Age2 INT NULL,
	AvgAge INT NULL,
	Distance INT NULL
GO

-- Update the Distance according to the event type.

UPDATE DataSet4Modified
SET Distance = 42, EveTyp = 6 
WHERE EVENT_TYPE = 'Marathon' AND LOCATION = 'MONTREAL QC'
GO

UPDATE DataSet4Modified
SET Distance = 42,EveTyp = 5
WHERE EVENT_TYPE IN ('Marathon','Challenge Marathon','Scotiabank Full Marathon',
'42.2km','42 KM Solo' ) AND LOCATION <> 'MONTREAL QC'
GO
UPDATE DataSet4Modified
SET Distance = 30,EveTyp = 4
WHERE EVENT_TYPE = '30K Marathon'
GO
UPDATE DataSet4Modified
SET Distance = 21 ,EveTyp = 3
WHERE EVENT_TYPE IN('Demi-Marathon','Half-Marathon','Half-Marathon - Demi-Marathon')
GO

UPDATE DataSet4Modified
SET Distance = 10,EveTyp = 2
WHERE EVENT_TYPE = '10K Marathon'
GO
UPDATE DataSet4Modified
SET Distance = 5,EveTyp = 1
WHERE EVENT_TYPE IN ('5K Road Race','5K')
GO


---------------------------------------------------------------------------------
--Update age range mentioned in the category excluding incorrect data
UPDATE DataSet4Modified
SET Age1 = CAST(SUBSTRING(LTRIM(RTRIM(CATEGORY)),2,2) as INT), 
Age2 = CAST(SUBSTRING(LTRIM(RTRIM(CATEGORY)),5,2) as INT)
WHERE LEFT(CATEGORY,1) IN ('M','F') 
and CATEGORY NOT IN ('MALE','FEMALE','M-JUN','F-SENIOR I','M-SEN II',
'M-SENIOR II','ATHENA',
'CLYDE','F-COMBO','Female 40-49','Femme','FEMMES','FILLES 4','F-SEN I','F-SEN II',
'F-SENIOR I','F-SENIOR II','F-U19','F-VET I','F-VET II','F-VETERAN II',
'GARCONS 8','HOMMES','M---11','MAÓTRE M40-49','MAÓTRE M50-59','MAITRE F30-39','MALE',
'Male 30-39','M-COMBO','M-ELITE','Mixte','M-JUN','M-Maitre','M-MAITRES','MPRO','M-PRO',
'M-SEN','M-SEN I','M-SEN II','M-SEN-II','M-SENIOR II','M-U23','M-VET I','M-VET II',
'M-VET III','M-Veteran','M-VETERAN I','M-VETERAN II','M-VETERAN III','NO AGE','NO-AGE',
'PARTICIPATIVE','PAS D''AGE','SENIOR M19-29','U0-0','F-','F5-','M6-','M7+','M7-8','M8-12')

GO

UPDATE DataSet4Modified
SET Age1 = Age2
WHERE Age1 = 0 AND Age2 <>0

UPDATE DataSet4Modified
SET Age2 = Age1
WHERE Age2 = 0 AND Age1 <>0


UPDATE DataSet4Modified
SET Age1 = 0,
	Age2 = 0
WHERE Age1 IS NULL
AND Age2 IS NULL

GO
-- Take average of the age ranges as the final age of the participant.
UPDATE DataSet4Modified
SET AvgAge = ( Age1+Age2)/2

GO
--------------------------------------------------------------------------

-- Update the given time in seconds 
UPDATE DataSet4Modified
SET 
	TimeInSeconds = DATEDIFF(second, 0, CAST([TIME] as TIME(0)))
WHERE [TIME] <> '-1'

-- Update the timeperkm (timeinseconds/Distance)
GO

UPDATE DataSet4Modified
SET TimePerKm = TimeInSeconds/Distance
WHERE Distance <>0

GO

UPDATE DataSet4Modified
SET TimePerKm = TimeInSeconds/Distance
WHERE EVENT_TYPE IN (
'51 KM Classique',
'45 KM STYLE LIBRE',
'50km Classique',
'40km Classique',
'40 Km Velo de Montagne',
'42 km',
'45 km',
'45km Classique'
)
AND TIME<>'-1'

GO

-- Update Distance for specific cases(manual updates)
UPDATE DataSet4Modified
SET Distance = 5
WHERE EVENT_TYPE IN ('5 km route',
'5 Km- Physio Sante',
'5 km - Course',
'5km Course et Marche',
'5 km - Buropro',
'5K Run - Course de 5K',
'5 km Course/Marche',
'5 km (Course)',
'5K Course',
'5 km Pneu Patry',
'5 km Marche',
'5 km Course et Marche',
'5 km Funnybone',
'5km Raquette',
'5km Marche',
'5.5 km',
'5km Run',
'5 KM FAMILIPRIX',
'Johnson 5km',
'5 km Run and Walk',
'Nage 5 km',
'5 km Run-Walk',
'5 km Run',
'5 km (vendredi)',
'5 km Raquette',
'5K Run',
'5 km - Canicourse',
'Course 5 km',
'5 km Media Challenge',
'5 km Poussettes'
)

GO

UPDATE DataSet4Modified
SET TimePerKm = TimeInSeconds/Distance
WHERE EVENT_TYPE IN ('5 km route',
'5 Km- Physio Sante',
'5 km - Course',
'5km Course et Marche',
'5 km - Buropro',
'5K Run - Course de 5K',
'5 km Course/Marche',
'5 km (Course)',
'5K Course',
'5 km Pneu Patry',
'5 km Marche',
'5 km Course et Marche',
'5 km Funnybone',
'5km Raquette',
'5km Marche',
'5.5 km',
'5km Run',
'5 KM FAMILIPRIX',
'Johnson 5km',
'5 km Run and Walk',
'Nage 5 km',
'5 km Run-Walk',
'5 km Run',
'5 km (vendredi)',
'5 km Raquette',
'5K Run',
'5 km - Canicourse',
'Course 5 km',
'5 km Media Challenge',
'5 km Poussettes'
)
AND TIME<>'-1'

GO
-----------------------------------------------------------------------------

--The final Modified dataset for linear regression using the mentioned features
USE ML
GO

SELECT *
INTO ML..Dataset4UpdatedNew 
FROM ML..DataSet4Modified

GO


SELECT PARTICIPANT_ID,AGE,GENDER,
CAST(ROUND(AvgTimeForMontrealMArathon2012,2) as NUMERIC(32,2)) AvgTimeForAllMarathons2012,
CAST(ROUND(AvgTimeForMontrealMArathon2013,2) as NUMERIC(32,2)) AvgTimeForAllMarathons2013,
CAST(ROUND(AvgTimeForMontrealMArathon2014,2) as NUMERIC(32,2)) AvgTimeForAllMarathons2014,
CAST(ROUND(AvgTimeInAllMarathons,2) as NUMERIC(32,2)) AvgTimeInAllMarathons,
CAST(ROUND(AvgTimeInAllEvents,2) as NUMERIC(32,2))  AvgTimeInAllEvents,
TotalNoOfMarathonEvents,
CAST(ROUND(AvgTimeForMontrealMArathon2015,2) as NUMERIC(32,2)) AvgTimeForAllMarathons2015
INTO RegressionData1
FROM (
SELECT PARTICIPANT_ID,AVG(AvgAge) as AGE,
AVG(CASE WHEN GENDER = 'M' THEN 1 
		 WHEN GENDER = 'F' THEN 0 ELSE -1 END) as GENDER,
AVG(CASE WHEN Evetyp IN (5,6) AND YEAR = '2012' THEN TimePerKm  END) as AvgTimeForMontrealMArathon2012,
AVG(CASE WHEN Evetyp IN (5,6) AND YEAR = '2013' THEN TimePerKm  END) as AvgTimeForMontrealMArathon2013,
AVG(CASE WHEN Evetyp IN (5,6) AND YEAR = '2014' THEN TimePerKm  END) as AvgTimeForMontrealMArathon2014,
AVG(CASE WHEN Evetyp IN (5,6) THEN TimePerKm  END) as AvgTimeInAllMarathons,
AVG(CASE WHEN Timeperkm <> 0 THEN TimePerKm  END) as AvgTimeInAllEvents,
SUM(CASE WHEN Evetyp IN (5,6) THEN 1 ELSE 0 END) as TotalNoOfMarathonEvents,
AVG(CASE WHEN Evetyp IN (5,6) AND YEAR = '2015' THEN TimePerKm  END) as AvgTimeForMontrealMArathon2015
FROM Dataset4UpdatedNew
GROUP BY PARTICIPANT_ID
)T
ORDER BY T.PARTICIPANT_ID

GO
--For cases where participant has not participated in the marathon for a particular year
-- instead of using the value 0, we have calculated the mean time of all participants 
-- for that year and used that for the final data prediction.
DECLARE @Avg2012 INT, @Avg2013 INT,@Avg2014 INT, @Avg2015 INT, @AvgAllMarathons INT, @AvgAllEvents INT

SELECT @Avg2012 = AVG(AvgTimeForAllMarathons2012),
@Avg2013 = AVG(AvgTimeForAllMarathons2013),
@Avg2014 = AVG(AvgTimeForAllMarathons2014),
@Avg2015 = AVG(AvgTimeForAllMarathons2015),
@AvgAllMarathons = AVG(AvgTimeInAllMarathons),
@AvgAllEvents = AVG(AvgTimeInAllEvents)
FROM RegressionData1

select @Avg2012,@Avg2013,@Avg2014,@Avg2015,@AvgAllMarathons,@AvgAllEvents

UPDATE RegressionData1
SET AvgTimeInAllMarathons = @AvgAllMarathons
WHERE AvgTimeInAllMarathons is null or AvgTimeInAllMarathons = 0

UPDATE RegressionData1
SET AvgTimeInAllEvents = @AvgAllEvents
WHERE AvgTimeInAllEvents is null or AvgTimeInAllEvents = 0

UPDATE RegressionData1
SET AvgTimeForAllMarathons2012 = @Avg2012
WHERE AvgTimeForAllMarathons2012 is null or AvgTimeForAllMarathons2012 = 0


UPDATE RegressionData1
SET AvgTimeForAllMarathons2013 = @Avg2013
WHERE AvgTimeForAllMarathons2013 is null or AvgTimeForAllMarathons2013 = 0


UPDATE RegressionData1
SET AvgTimeForAllMarathons2014 = @Avg2014
WHERE AvgTimeForAllMarathons2014 is null or AvgTimeForAllMarathons2014 = 0


UPDATE RegressionData1
SET AvgTimeForAllMarathons2015 = @Avg2015
WHERE AvgTimeForAllMarathons2015 is null or AvgTimeForAllMarathons2015 = 0

GO

SELECT * FROM RegressionData1

-- To generate the CSV, BCP (bulk copy) utility is used in sql server to serve this purpose
DECLARE @sql VARCHAR(8000)
select @sql = 'bcp "SELECT PARTICIPANT_ID,AGE,GENDER,AvgTimeForAllMarathons2012,AvgTimeForAllMarathons2013,AvgTimeForAllMarathons2014,AvgTimeInAllMarathons,AvgTimeInAllEvents,TotalNoOfMarathonEvents,AvgTimeForAllMarathons2015 FROM ML.dbo.RegressionData1" queryout C:\bcp\FinaldataNew.csv -c -t, -T -S ' + @@SERVERNAME

exec master..xp_cmdshell @sql

GO


-----------------------------------------------------------------------------
