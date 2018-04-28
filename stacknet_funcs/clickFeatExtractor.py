def do_next_prev_Click(predictors, df, agg_suffix, agg_type='float32'):
    """

    :param predictors: List of predictor variables
    :param df: Dataframe to do the aggregations on
    :param agg_suffix: Suffix to be added to end of variable name
    :param agg_type: Data type of value returned
    :return: Dataframe with new aggregations
    """

    import pandas as pd
    import gc

    print('Extracting new features...')

    GROUP_BY_NEXT_CLICKS = [

        # V3
        {'groupby': ['ip', 'app', 'device', 'os', 'channel']},
        {'groupby': ['ip', 'os', 'device']},
        {'groupby': ['ip', 'os', 'device', 'app']}
    ]

    # Calculate the time to next click for each group
    for spec in GROUP_BY_NEXT_CLICKS:

        # Name of new feature
        new_feature = '{}_{}'.format('_'.join(spec['groupby']), agg_suffix)

        # Unique list of features to select
        all_features = spec['groupby'] + ['click_time']

        # Run calculation

        if agg_suffix == "nextClick":
            df[new_feature] = (df[all_features].groupby(spec[
                                                            'groupby']).click_time.shift(
                -1) - df.click_time).dt.seconds.astype(agg_type)
        elif agg_suffix == "prevClick":
            df[new_feature] = (df.click_time - df[all_features].groupby(spec[
                                                                            'groupby']).click_time.shift(
                +1)).dt.seconds.astype(agg_type)
        predictors.append(new_feature)
        gc.collect()

    return predictors, df

def do_countuniq(predictors, df, group_cols, counted, agg_type='uint32', show_max=False, show_agg=True):
    """

    :param predictors: List of predictors
    :param df: Dataframe to be aggregated
    :param group_cols: Columns to be grouped by for aggregate features to be calculated
    :param counted: Variable that the unique count is being calculated for
    :param agg_type: Type of data outputted by aggregations
    :param show_max: Flag
    :param show_agg: Flag
    :return: Dataframe with aggregated columns calculated
    """

    import pandas as pd
    import gc

    agg_name = '{}_by_{}_countuniq'.format(('_'.join(group_cols)), (counted))
    if show_agg:
        print("\nCounting unqiue ", counted, " by ", group_cols, '... and saved in', agg_name)
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].nunique().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print(agg_name + " max value = ", df[agg_name].max())
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
    print('predictors %s' % (predictors))

    gc.collect()
    return predictors, df

def do_count(predictors, df, group_cols, agg_type='uint32', show_max=False, show_agg=True):

    import pandas as pd
    import gc

    agg_name = '{}count'.format('_'.join(group_cols))
    if show_agg:
        print("\nAggregating by ", group_cols, '... and saved in', agg_name)
    gp = df[group_cols][group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print(agg_name + " max value = ", df[agg_name].max())
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)

    gc.collect()
    return predictors, df

def do_cumcount(predictors, df, group_cols, counted, agg_type='uint32', show_max=False, show_agg=True):

    import pandas as pd
    import gc

    agg_name = '{}_by_{}_cumcount'.format(('_'.join(group_cols)), (counted))
    if show_agg:
        print("\nCumulative count by ", group_cols, '... and saved in', agg_name)
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].cumcount()
    df[agg_name] = gp.values
    del gp
    if show_max:
        print(agg_name + " max value = ", df[agg_name].max())
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
    #     print('predictors',predictors)
    gc.collect()
    return predictors, df

def do_mean(predictors, df, group_cols, counted, agg_type='float32', show_max=False, show_agg=True):

    import pandas as pd
    import gc

    agg_name = '{}_by_{}_mean'.format(('_'.join(group_cols)), (counted))
    if show_agg:
        print("\nCalculating mean of ", counted, " by ", group_cols, '... and saved in', agg_name)
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].mean().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print(agg_name + " max value = ", df[agg_name].max())
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
    print("predictors %s" % (predictors))
    gc.collect()
    return predictors, df


def do_var(predictors, df, group_cols, counted, agg_type='float32', show_max=False, show_agg=True):

    import pandas as pd
    import gc

    agg_name = '{}_by_{}_var'.format(('_'.join(group_cols)), (counted))
    if show_agg:
        print("\nCalculating variance of ", counted, " by ", group_cols, '... and saved in', agg_name)
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].var().reset_index().rename(columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print(agg_name + " max value = ", df[agg_name].max())
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
    print("predictors %s" % (predictors))

    gc.collect()

    return predictors, df