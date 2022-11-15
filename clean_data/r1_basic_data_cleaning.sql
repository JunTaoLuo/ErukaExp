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

update cleaned.property_transfer 
	set parcelid = upper(parcelid);

-- Formatting dates
update cleaned.property_transfer 
	set year_of_sale = '19' || year_of_sale
	where year_of_sale in ('98', '99');

alter table cleaned.property_transfer 
	add column date_of_sale DATE;

update cleaned.property_transfer 
	set date_of_sale = to_date(concat(year_of_sale, '-', month_of_sale, '-', day_of_sale),
								'YYYY-MM-DD');
							
-- Formatting property class code variable
update cleaned.property_transfer 
	set property_class = null
	where property_class = '   ';

alter table cleaned.property_transfer
	alter column property_class type int using(property_class::int);

-- Cleaning num_parcels_sold
update cleaned.property_transfer
	set num_parcels_sold = null
	where num_parcels_sold = '  ' or num_parcels_sold = ' ';

update cleaned.property_transfer 
	set num_parcels_sold = replace(num_parcels_sold, '"', '');

alter table cleaned.property_transfer 
	alter column num_parcels_sold type float using(num_parcels_sold::float);

update cleaned.property_transfer
	set num_parcels_sold = round(num_parcels_sold);

alter table cleaned.property_transfer 
	alter column num_parcels_sold type int using(num_parcels_sold::int);

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

-- Cleaning BuildingInfo
update cleaned.building_info 
	set "YEARBUILT" = null 
	where "YEARBUILT" = 0;
	
-- Cleaning historic sales
update cleaned.historic_sales 
	set year_built = null
	where year_built = 0;
							
-- Renaming all IDs to make them consistent (parcelid)
alter table cleaned.building_info 
	rename column "PARCELID" to parcelid;

alter table cleaned.building_info
	rename column "ATTIC_SQFT" to attic_sqft;

alter table cleaned.building_info
	rename column "BSMT_SQFT" to bsmt_sqft;

alter table cleaned.building_info
	rename column "LIVE_FSQFT" to live_sqft;

alter table cleaned.building_info
	rename column "SQFT" to sqft;

alter table cleaned.building_info
	rename column "SQFT_FLR1" to sqft_flr1;

alter table cleaned.building_info
	rename column "SQFT_FLR2" to sqft_flr2;

alter table cleaned.building_info
	rename column "SQFT_FLRH" to sqft_flrh;

alter table cleaned.building_info
	rename column "STORYHT" to story_ht;
	
alter table cleaned.building_info
	rename column "YEARBUILT" to year_built;

update cleaned.building_info 
	set parcelid = upper(parcelid);

alter table cleaned.monthly_tax 
	rename column parcel_number to parcelid;
	
alter table cleaned.historic_sales 
	rename column parcel_number to parcelid;
	
alter table cleaned.rental_registration 
	rename column raw_parcel_number to parcelid;

-- Creating indices on parcelid for better merging
alter table cleaned.monthly_tax add primary key (parcelid);
alter table cleaned.historic_sales add primary key (parcelid);
alter table cleaned.rental_registration add primary key (parcelid);
create index on cleaned.property_transfer(parcelid);
create index on cleaned.building_info(parcelid);
