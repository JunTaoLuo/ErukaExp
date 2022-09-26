-- Table/schema setup
drop table if exists cleaned.building_info;
drop table if exists cleaned.historic_sales;
drop table if exists cleaned.monthly_tax;
drop table if exists cleaned.property_transfer;
drop table if exists cleaned.rental_registration;

drop schema if exists cleaned;

create schema cleaned;

create table cleaned.building_info as (select * from "BuildingInfo");
create table cleaned.historic_sales as (select * from "HistoricSales");
create table cleaned.monthly_tax as (select * from "MonthlyTax");
create table cleaned.property_transfer as (select * from "PropertyTransfer");
create table cleaned.rental_registration as (select * from "RentalRegistration");

-- Cleaning property transfers
delete from cleaned.property_transfer 
where year_of_sale = 'YearSale'; -- dropping a dummy row

update cleaned.property_transfer 
	set book = replace(book, '*', ''); 
-- replace the * doesn't lose any info; seems like * was used to indicate if
-- the property was new; all cases where there is a *, the land/building value say 'New'
-- creating a new field to capture this info below (after parcelid is created)

alter table cleaned.property_transfer
	add column parcelid char(13);

update cleaned.property_transfer 
	set parcelid = lpad(book, 3, '0') || lpad(plat, 4, '0') || lpad(parcel, 4, '0') || lpad(multi_owner, 2, '0');

-- Formatting dates
update cleaned.property_transfer 
	set year_of_sale = '19' || year_of_sale
	where year_of_sale in ('98', '99');

alter table cleaned.property_transfer 
	add column date_of_sale DATE;

update cleaned.property_transfer 
	set date_of_sale = to_date(concat(year_of_sale, '-', month_of_sale, '-', day_of_sale),
								'YYYY-MM-DD');

-- Creating is-new flag for property transfer (right now stored in land_value)
alter table cleaned.property_transfer 
	add column is_new int;

update cleaned.property_transfer 
	set is_new = case when land_value like '%New%' then 1 else 0 end;
							
-- Formatting variables that should be numeric as numeric
update cleaned.property_transfer 
	set land_value = null, building_value = null
	where land_value ilike '%New%' or building_value ilike '%New%';

alter table cleaned.property_transfer 
	alter column land_value type double precision using(land_value::double precision),
	alter column building_value type double precision using(building_value::double precision),
	alter column sale_price type double precision using(sale_price::double precision);
							
-- Renaming all IDs to make them consistent (parcelid)
alter table cleaned.building_info 
	rename column PARCELID to parcelid;

alter table cleaned.monthly_tax 
	rename column parcel_number to parcelid;
	
alter table cleaned.historic_sales 
	rename column parcel_number to parcelid;
	
alter table cleaned.rental_registration 
	rename column raw_parcel_number to parcelid;

-- Todo: think about duplicate IDs in BuildingInfo

-- Todo: Declare parcelid as either index or primary key in data
-- Creating indices on parcelid for better merging
alter table cleaned.monthly_tax add primary key (parcelid);
alter table cleaned.historic_sales add primary key (parcelid);
alter table cleaned.rental_registration add primary key (parcelid);
create index on cleaned.property_transfer(parcelid);
create index on cleaned.building_info(parcelid);
