from memo import memo


@memo
def get_edit_dis(str1, str2):
    if len(str1) == 0: return len(str2)
    if len(str2) == 0: return len(str1)

    return min(
        [get_edit_dis(str1[:-1], str2) + 1,
         get_edit_dis(str1, str2[:-1]) + 1,
         get_edit_dis(str1[:-1], str2[:-1]) + (0 if str1[-1] == str2[-1] else 2)]
    )


def get_edit_dis2(str1, str2):
    m = len(str1)
    n = len(str2)
    dp = [[0 for i in range(n + 1)] for j in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):

            if i == 0:
                dp[i][j] = j

            elif j == 0:
                dp[i][j] = i

            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]

            else:
                dp[i][j] = min(dp[i][j - 1] + 1,  # Insert
                               dp[i - 1][j] + 1,  # Remove
                               dp[i - 1][j - 1] + 2)  # Replace

    return dp[m][n]


str1 = "beijing abc"
str2 = "biejing"

d = get_edit_dis(str1, str2)
print('edit dis:', d)

d = get_edit_dis2(str1, str2)
print('edit dis:', d)
