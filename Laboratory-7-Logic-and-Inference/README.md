# Laboratory 7 - Logic and Inference

This README provides instructions on how to run the Prolog date manipulation code using the SWI-Prolog online environment, SWISH. This platform allows you to execute Prolog code directly in your browser without any installation.

## Prerequisites

Ensure you have:

- Internet access: A stable internet connection is required to access the SWI-Prolog web interface.
- A modern web browser: Such as Google Chrome, Firefox, Safari, or Edge.

## Setup and Execution

1. Access SWI-Prolog Online - Go to the SWI-Prolog online editor and interpreter. You can access it via this URL: SWISH: SWI-Prolog for SHaring.

2. Create a New Program - Once you are on the SWISH website, start a new program by clicking on the Create a new program button or similar option available on the site.

3. Copy and Paste Your Code - Copy the Prolog date manipulation code from your local file. Then paste the code into the online editor in the SWISH interface.

4. Run Your Code
   After pasting your code, you can run it directly in the browser

   Click the Run button or type your query directly in the provided command line interface at the bottom of the editor.

   For example, to manipulate dates, you can use the `add_date` and `sub_date` predicates with the desired input parameters.

   **Examples**

   ```prolog
   ?- add_date("2205", 14)   % add 14 days from 22nd of May
   "0506"  % 5th of June
   ?- sub_date("2205", 10)   % subtract 10 days from 22nd of May
   "1205"  % 12th of May
   ```

## Additional tests

```prolog
?- add_date("1501", 30)
"1402"  % 14th of February
```

```prolog
?- add_date("2802", 90)
"2905"  % 29th of May
```

```prolog
?- sub_date("0503", 20)
"1402"  % 13th of February
```

```prolog
?- sub_date("0108", 75)
"1805"  % 17th of May
```
