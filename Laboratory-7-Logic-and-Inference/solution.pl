string_to_date(Date, Day, Month) :-
    nonvar(Date),
    months(Months),
    nth1(Month_number, Months, Month),
    sub_string(Date, 0, 2, _, Day_string),
    atom_number(Day_string, Day),
    sub_string(Date, 2, 2, _, Month_string),
    atom_number(Month_string, Month_number).

string_to_date(Date, Day, Month) :-
    nonvar(Day), nonvar(Month),
    months(Months),
    nth1(Month_number, Months, Month),
    format(string(Date),'~`0t~d~2+~`0t~d~2+', [Day, Month_number]).

months(Months) :- Months = [january, february, march, april,
                             may, june, july, august, september,
                             october, november, december].

find_months_before(Month, Months_before) :-
    months(Y),
    append(B, _, Y),
    last(B, Month),
    select(Month, B, Months_before).

months_to_days(Months, Days_of_month) :-
    findall(Z, (
               select(Month, Months, _),
               month_days(Month, true, Z)
               ), Days_of_month).

date_month_to_month_start(Month, Month_start) :-
    find_months_before(Month, Months),
    months_to_days(Months, Days_of_month),
    sum_list(Days_of_month, Month_start).

date_to_day_of_year(Day, Month, Day_of_year) :-
    nonvar(Day), nonvar(Month),
    date_month_to_month_start(Month, Month_start),
    Day_of_year is Month_start + Day.

date_to_day_of_year(Day, Month, Day_of_year) :-
    nonvar(Day_of_year),
    date_month_to_month_start(Next_month, Next_month_start),
    Day_of_year < Next_month_start,
    months(Months),
    nextto(Month, Next_month, Months),
    date_month_to_month_start(Month, Month_start),
    Day is Day_of_year - Month_start.

is_leap_year(Year) :-
    0 is Year mod 4,
    not(0 is Year mod 100).

is_leap_year(Year) :-
    0 is Year mod 400.

leap_dependent(february).
long_month(january).
long_month(march).
long_month(may).
long_month(july).
long_month(august).
long_month(october).
long_month(december).

short_month(april).
short_month(june).
short_month(september).
short_month(november).

month_days(february, true, X) :- X = 29.
month_days(february, false, X) :- X = 28.
month_days(Month, _, X) :- long_month(Month), X = 31.
month_days(Month, _, X) :- short_month(Month), X = 30.

add_date(Date, Days) :- 
    string_to_date(Date, Day, Month),
    date_to_day_of_year(Day, Month, Day_of_year),
    M_day_of_year is Day_of_year + Days,
    date_to_day_of_year(M_day, M_month, M_day_of_year),
    string_to_date(M_date, M_day, M_month),
    write(M_date).

sub_date(Date, Days) :- 
    string_to_date(Date, Day, Month),
    date_to_day_of_year(Day, Month, Day_of_year),
    M_day_of_year is Day_of_year - Days,
    date_to_day_of_year(M_day, M_month, M_day_of_year),
    string_to_date(M_date, M_day, M_month),
    write(M_date).