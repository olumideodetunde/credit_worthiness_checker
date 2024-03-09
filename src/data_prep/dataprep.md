From literature review, the following features have been selected to build a simple dataset
1.Age - found in static_data_2 - column name: dateofbirth_337D, dateofbirth_342D
2.Main income amount -  found in static data - column name: maininc_215A & found in person_data - column name: mainoccupationinc_384A & found in previous_application_df - column name: mainoccupationinc_437A
3.Marital status - found in previous_application_df - column name: familystate_726L & found in person_data - column name: familystate_447L
4.Gender - found in person_data - column: gender_992L & found in person_data - column: sex_738L
6.number of children - found in previous_application_df - column name:childnum_21L  & person_data - column name: childnum_185L
7.existing previous credit status if any - found in previous_application_df - column name: credacc_status_367L 
8.Existing debt amount - found in previous_application_df - column name: outstandingdebt_522A

other adhoc column names:
    - birth_259D,Date of birth of the person.
    - birthdate_574D,Client's date of birth (credit bureau data).
    - birthdate_87D,Birth date of the person.