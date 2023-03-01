drop table if exists processed.building_info;
drop table if exists processed.historic_sales;
drop table if exists processed.monthly_tax;
drop table if exists processed.property_transfer;
drop table if exists processed.rental_registration;

drop schema if exists processed;

create schema processed;

-- creating tables only choosing relevant prop class codes

-- Join only keeps the buildings that merge into monthlytax
create table processed.building_info as 
	(select bi.*, mt.prop_class_code, mt.class_description 
	from cleaned.building_info bi join cleaned.monthly_tax mt on bi.parcelid = mt.parcelid
	where mt.prop_class_code in ({class_codes}) 
	);

-- Join only keeps the parcels that merge into monthlytax
create table processed.historic_sales as 
	(select hs.*, mt.prop_class_code, mt.class_description, mt.area_description
	 from cleaned.historic_sales hs join cleaned.monthly_tax mt on hs.parcelid = mt.parcelid
	 where mt.prop_class_code in ({class_codes})
	 );
create table processed.monthly_tax as (select * from cleaned.monthly_tax);
create table processed.property_transfer as (select * from cleaned.property_transfer);
create table processed.rental_registration as (select * from cleaned.rental_registration);

-- Deal with duplicates in parcelid by assigning a building id (to be reshaped based on this ID in Python script)
alter table processed.building_info
	add column building_id int,
	add column tot_buildings int;

-- Note: ctid is a background auto-generated uid for each row
update processed.building_info bi
	set building_id = building_id2
		from (select ctid, row_number() over (partition by parcelid) as building_id2
			  from processed.building_info) bi2
	where bi.ctid = bi2.ctid;


-- creating a total_buildings field (# of distinct buildings on that parcel)
update processed.building_info bi
	set tot_buildings = tot_buildings2
		from (select parcelid, count(*) as tot_buildings2
			  from processed.building_info
			  group by parcelid) bi2
	where bi.parcelid = bi2.parcelid;

-- Filling in live_sqft = sum of all other square footage if sum > 0 and live_fsqft is zero
update processed.building_info bi
	set live_sqft = (sqft_flr1 + sqft_flr2 + attic_sqft) -- note: this logic holds for most of the data
	where bi.live_sqft = 0;

-- Removing rows that are not secondary/tertiary buildings on the parcel, and have zero on all columns (likely torn down structures)
delete from processed.building_info
	where (live_sqft + attic_sqft + sqft_flr1 + sqft_flr2 + sqft_flrh) = 0
	and tot_buildings = 1;

-- Dropping unnecessary columns in the datasets
alter table processed.building_info
	drop column sqft;


-- Creating columns that will be used in the analysis

-- Grouping appraisal area by collapsing subtypes
alter table processed.historic_sales
	add column area_description_grouped varchar(350);
update processed.historic_sales
	set area_description_grouped = area_description;

-- Keeping only things to the left of the parenthesis
update processed.historic_sales
	set area_description_grouped = regexp_replace(area_description_grouped, '(?<=\().*', '', 'g');

-- Removing numbers and the opening parenthesis
update processed.historic_sales
	set area_description_grouped = regexp_replace(area_description_grouped, '[\d\()]', '', 'g');

-- Removing double blank spaces
update processed.historic_sales
	set area_description_grouped = regexp_replace(area_description_grouped, '(\s{2,})', '', 'g');

-- Trimming to finish
update processed.historic_sales
	set area_description_grouped = trim(area_description_grouped);

-- Making a livable sqft proportion
alter table processed.building_info
	add column live_sqft_prop double precision;
update processed.building_info bi
	set live_sqft_prop = live_sqft::double precision/(attic_sqft+bsmt_sqft+sqft_flr1+sqft_flr2+sqft_flrh);

-- Combine garage type categories (removing + sign)
alter table processed.historic_sales
	add column garage_type_grouped varchar(250);
update processed.historic_sales
	set garage_type_grouped = garage_type;

update processed.historic_sales
	set garage_type_grouped = replace(garage_type_grouped, '+', '');

update processed.historic_sales
	set garage_type_grouped = TRIM(garage_type_grouped);

-- Grouping values in basement
alter table processed.historic_sales
	add column basement_grouped varchar(200);

update processed.historic_sales
	set basement_grouped = 'Crawl Space'
where basement = 'Crawl Space';

update processed.historic_sales
	set basement_grouped = 'Basement'
where basement != 'Crawl Space' and basement is not null;

-- Grouping values in grade
alter table processed.historic_sales
	add column grade_grouped varchar(200),
	add column grade_numeric int;

update processed.historic_sales
	set grade_grouped = grade
where grade is not null;

update processed.historic_sales
	set grade_grouped = 'Exceptional'
where grade_grouped in ('Outstanding', 'Exceptional', 'Exceptional+', 'Extraordinary');

update processed.historic_sales
	set grade_grouped = 'Excellent'
where grade_grouped in ('Excellent', 'Excellent+');

-- Creating a numeric variable for grade
update processed.historic_sales
	set grade_numeric = 9
where grade_grouped = 'Exceptional';

update processed.historic_sales
	set grade_numeric = 8
where grade_grouped = 'Excellent';

update processed.historic_sales
	set grade_numeric = 7
where grade_grouped = 'Very Good';

update processed.historic_sales
	set grade_numeric = 6
where grade_grouped = 'Good';

update processed.historic_sales
	set grade_numeric = 5
where grade_grouped = 'Average';

update processed.historic_sales
	set grade_numeric = 3
where grade_grouped = 'Fair';

update processed.historic_sales
	set grade_numeric = 1
where grade_grouped = 'Poor';
