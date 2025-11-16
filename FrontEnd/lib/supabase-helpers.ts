'use client'

import { createClient } from '@supabase/supabase-js'

export type TryOnHistoryItem = {
  id?: string
  user_id?: string
  product_id: number
  product_name: string
  product_image: string
  user_image_url: string
  cloth_image_url: string
  result_image_url: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  error?: string
  created_at?: string
  completed_at?: string
}

let supabaseClient: ReturnType<typeof createClient> | null = null

// Helper function to get the Supabase client on the client side
export function getSupabaseClient() {
  if (typeof window === 'undefined') {
    console.log('Cannot initialize Supabase on server side')
    return null
  }
  
  if (!supabaseClient) {
    const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
    const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
    
    if (!supabaseUrl || !supabaseKey) {
      console.error('Missing Supabase environment variables')
      return null
    }
    
    supabaseClient = createClient(supabaseUrl, supabaseKey)
    console.log('Supabase client initialized')
  }
  
  return supabaseClient
}

// Function to save try-on history
export async function saveToHistory(item: TryOnHistoryItem): Promise<{ success: boolean, error?: any }> {
  try {
    const client = getSupabaseClient()
    if (!client) {
      console.error('Supabase client not available')
      return { success: false, error: 'Client not available' }
    }
    
    // Prepare the data to save
    const dataToSave = {
      ...item,
      created_at: new Date().toISOString()
    }
    
    console.log('Saving try-on history:', dataToSave)
    
    const { error } = await client
      .from('try_on_history')
      .insert([dataToSave])
    
    if (error) {
      console.error('Error saving try-on history:', error)
      return { success: false, error }
    }
    
    console.log('Try-on history saved successfully')
    return { success: true }
  } catch (err) {
    console.error('Exception saving try-on history:', err)
    return { success: false, error: err }
  }
}

// Function to get try-on history
export async function getHistory(options?: {
  status?: string;
  limit?: number;
  offset?: number;
}): Promise<TryOnHistoryItem[]> {
  try {
    const client = getSupabaseClient()
    if (!client) {
      console.error('Supabase client not available')
      return []
    }
    
    let query = client
      .from('try_on_history')
      .select('*')
      .order('created_at', { ascending: false })
    
    // Apply status filter if provided
    if (options?.status && options.status !== 'all') {
      query = query.eq('status', options.status)
    }
    
    // Apply pagination if provided
    if (options?.limit) {
      query = query.limit(options.limit)
    } else {
      query = query.limit(50) // Default limit
    }
    
    if (options?.offset) {
      query = query.range(options.offset, options.offset + (options.limit || 50) - 1)
    }
    
    const { data, error } = await query
    
    if (error) {
      console.error('Error fetching try-on history:', error)
      return []
    }
    
    return data || []
  } catch (err) {
    console.error('Exception fetching try-on history:', err)
    return []
  }
}

// Function to delete a try-on history item
export async function deleteHistoryItem(id: string): Promise<boolean> {
  try {
    const client = getSupabaseClient();
    if (!client) {
      console.error('Supabase client not available');
      return false;
    }
    
    console.log(`Attempting to delete history item with ID: ${id}`);
    
    const { error } = await client
      .from('try_on_history')
      .delete()
      .eq('id', id);
    
    if (error) {
      console.error('Error deleting history item:', error);
      return false;
    }
    
    console.log('History item deleted successfully');
    return true;
  } catch (err) {
    console.error('Exception when deleting history item:', err);
    return false;
  }
}

// Function to delete multiple try-on history items
export async function bulkDeleteHistoryItems(ids: string[]): Promise<{ success: boolean, deletedCount: number }> {
  try {
    const client = getSupabaseClient();
    if (!client) {
      console.error('Supabase client not available');
      return { success: false, deletedCount: 0 };
    }
    
    if (!ids.length) {
      return { success: true, deletedCount: 0 };
    }
    
    console.log(`Attempting to delete ${ids.length} history items`);
    
    const { data, error } = await client
      .from('try_on_history')
      .delete()
      .in('id', ids)
      .select('id');
    
    if (error) {
      console.error('Error bulk deleting history items:', error);
      return { success: false, deletedCount: 0 };
    }
    
    const deletedCount = data?.length || 0;
    console.log(`Successfully deleted ${deletedCount} history items`);
    return { success: true, deletedCount };
  } catch (err) {
    console.error('Exception when bulk deleting history items:', err);
    return { success: false, deletedCount: 0 };
  }
}

// Function to get history count by status
export async function getHistoryStats(): Promise<{ 
  total: number; 
  completed: number; 
  pending: number; 
  processing: number; 
  failed: number; 
}> {
  try {
    const client = getSupabaseClient();
    if (!client) {
      console.error('Supabase client not available');
      return { total: 0, completed: 0, pending: 0, processing: 0, failed: 0 };
    }
    
    const { data, error } = await client
      .from('try_on_history')
      .select('status', { count: 'exact' });
    
    if (error) {
      console.error('Error fetching history stats:', error);
      return { total: 0, completed: 0, pending: 0, processing: 0, failed: 0 };
    }
    
    const total = data?.length || 0;
    const completed = data?.filter(item => item.status === 'completed').length || 0;
    const pending = data?.filter(item => item.status === 'pending').length || 0;
    const processing = data?.filter(item => item.status === 'processing').length || 0;
    const failed = data?.filter(item => item.status === 'failed').length || 0;
    
    return { 
      total, 
      completed, 
      pending, 
      processing, 
      failed 
    };
  } catch (err) {
    console.error('Exception fetching history stats:', err);
    return { total: 0, completed: 0, pending: 0, processing: 0, failed: 0 };
  }
}
