-- what do these tables look like?
select * from "BuildingInfo" bi limit 300;

select * from "HistoricSales" hs limit 300;

select * from "MonthlyTax" mt limit 300;

select * from "PropertyTransfer" pt limit 300;

-- how do we identify pre-1998 sales in the data? What date/time variables exist in the schema?
select * from information_schema.columns
where table_schema = 'public'
and column_name ilike '%date%' or column_name ilike '%year%';

--understanding date-of-sale in historic sales
select parcel_number, date_of_sale from "HistoricSales" hs
where parcel_number in 
	(select parcel_number from "HistoricSales" 
	 group by parcel_number having count(parcel_number) > 1);
-- Note: there are no duplicate parcel_numbers: this data is at a parcel_number level
	
-- how complete is date-of-sale?
select count(*) from "HistoricSales" hs where date_of_sale is null;

select min(date_of_sale) from "HistoricSales" hs; -- earliest date is 1900-01-01

-- sales before 1998
select count(*) from "HistoricSales" hs where date_of_sale < '1998-01-01';

--total parcel_numbers in historicsales data
select count(*) from "HistoricSales" hs;

-- distribution of how many parcels come under a parcel number
select number_of_parcels, count(*) from "HistoricSales" hs 
group by number_of_parcels
order by number_of_parcels;

-- earliest sale in propertytransfer data is 1998-01-02 (need to format/clean dates to improve this query)
select * from "PropertyTransfer" pt where year_of_sale = '98' and month_of_sale = '1' and day_of_sale = '2'

-- Monthly tax transfer_date to identify properties sold before 1998

-- how many unique parcel_numbers in the monthly tax data?
select count(distinct parcel_number) from "MonthlyTax" mt;
select count(*) from "MonthlyTax" mt; 
-- parcel_number is again, the unique ID in this dataset

-- browsing monthly tax records for one parcel
select * from "MonthlyTax" mt where parcel_number = '0010001017400';

-- Earliest and latest transfer_date in monthly tax
select min(transfer_date), max(transfer_date) from "MonthlyTax" mt;
-- earliest is 12/30/1899, latest is 2022-08-04

select count(*) from "MonthlyTax" mt where transfer_date is null;

select count(*) from "MonthlyTax" mt where transfer_date < '1998-01-01';

-- Checking merge between the MonthlyTax and HistoricSales
select count(*)
from "MonthlyTax" mt left outer join "HistoricSales" hs on mt.parcel_number = hs.parcel_number 
where hs.parcel_number is null;

select count(*)
from "HistoricSales" hs left outer join "MonthlyTax" mt on mt.parcel_number = hs.parcel_number
where mt.parcel_number is null;

-- BuildingInfo exploration
select count(*) from "BuildingInfo" bi;
select count(distinct "PARCELID") from "BuildingInfo" bi;
-- there are some duplicates in parcelid

-- Browsing/exploring the duplicates
select * from "BuildingInfo" bi
where "PARCELID" in 
	(select "PARCELID" from "BuildingInfo"  
	 group by "PARCELID" having count("PARCELID") > 1)
order by "PARCELID";

-- Checking merge between BuildingInfo and other datasets
select count(*)
from "BuildingInfo" bi left outer join "HistoricSales" hs on bi."PARCELID" = hs.parcel_number 
where hs.parcel_number is null;

select count(*)
from "BuildingInfo" bi left outer join "MonthlyTax" mt on bi."PARCELID" = mt.parcel_number 
where mt.parcel_number is null;





